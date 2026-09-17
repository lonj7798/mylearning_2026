<!-- chapter: ch-36
     track: sft
     kind: lab
     title: Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split
     deps: [ch-35a, ch-30a]
     sources: [[tulu-3]], [[tulu-3-lab-evidence]], [[tulu-3-sft-mix]], [[loss-masking-prompt]], [[lora-learns-less-forgets-less]],
              [[tulu-1-how-far-can-camels-go]], [[benchmark-variance-quantified]], [[adding-error-bars-evals]], [[ifeval]], [[ifbench]],
              [[open-instruct-allenai-recipes]], [[open-instruct-allenai-recipes-recipe]], [[packing-position-ids]],
              [[hf-chat-template-assistant-mask]], [[karpathy-training-neural-net-recipe]], [[signal-and-noise-eval]], [[judge-llm-bias]]
     figures: figures/forgetting-ci-planner.html
     revised: 2026-09 (generality revision)
-->

# Chapter 36 — Lab: SFT Run with Masking Tests, a Forgetting Report, and a Held-Out Evaluation Split

> **Core insight.** An SFT run can raise its target score while lowering abilities it did not train. On LLaMA 13B, the 20,022-example Code-Alpaca set raised Codex-Eval (HumanEval pass@10) from 28.6 to 34.2 but lowered BBH from 39.3 to 35.6 and TyDiQA from 43.2 to 38.9; a larger seven-dataset mixture (490,694 examples, the sum of its datasets' Table 1 counts; not size-matched) scored higher than Code-Alpaca on all six evaluations ([[tulu-1-how-far-can-camels-go]] Tables 1 and 3). Gains on a development suite can also fail to transfer: removing Tülu 3's Persona datasets (math, coding, and a Precise IF set built from IFEval's constraint types) lowered IFEval from 72.8 to 53.6 while IFEval-OOD moved from 17.6 to 18.0 ([[tulu-3-lab-evidence]], Table 32). This lab gates training on unit tests of what the loss is computed on, and judges each run by paired base-versus-SFT differences on suites the mixture does not target and on a held-out suite that no decision uses, with seed variance included in every interval.
>
> **Guideline.** When an SFT run will be compared with its base model or with another run, run template, label, and packing unit tests before the first optimizer step, because a chat template without a `{% generation %}` block makes `apply_chat_template` return an all-zero assistant mask with only a logged warning ([[hf-chat-template-assistant-mask]], transformers v4.46.0 L1805-1808). When choosing data, epochs, or LoRA settings, decide on a development suite and keep a held-out suite with a different benchmark per skill unexamined until the decisions are written down, because Tülu 3's instruction-following data choices overfit IFEval ([[tulu-3-lab-evidence]], §7.4.1). When reporting a difference, compute a paired 95% interval over items and over at least 3 SFT seeds, because Tülu 3 70B SFT seed averages ranged from 70.0 to 72.6 (same source, Table 14) and a 541-prompt IFEval score near 60% has an item-level half-width of 4.1 points (derived with the formula in [[benchmark-variance-quantified]] §3.1). When compute allows only a 1B model, keep the data-arm axis, the epoch axis, and the full forgetting report, and drop the loss and LoRA axes, because in the studies of §3 and §4.1 data composition and training duration changed scores on abilities the data did not target ([[tulu-1-how-far-can-camels-go]] Table 3; [[lora-learns-less-forgets-less]] Table S6).

## Why this chapter matters for a general-purpose model

The training pipeline for a general-purpose model runs pre-training → mid-training → SFT → preference optimization → RL → evaluation. This lab is the last chapter of the SFT phase. The preference and RL phases that follow (from [[ch-15]]) start from an SFT checkpoint of the kind this lab produces, the evaluation lab [[ch-53]] reuses this lab's runs and generations, and the forgetting report uses the format defined in [[ch-30a]] §6.

The lab addresses three measurable problems:
1. **Errors in what is trained.** A wrong template, label mask, or packed attention pattern raises no exception and still produces a decreasing loss curve. Released pipelines contain such errors: OLMo 3 7B Think DPO and RL used a chat template "slightly different" from the SFT template because of "a minor miscommunication" (L22), and the open-instruct tokenization code "incorrectly masks the first <think> token as part of the prompt", which Think SFT tokenization avoids by using a template without `<think>` (L28, L39) ([[open-instruct-allenai-recipes]], `docs/olmo3.md` at commit 098424c).
2. **Forgetting.** SFT on a single instruction dataset can lower scores on abilities outside the data. In [[loss-masking-prompt]] Table 1, every response-only LLaMA-2-7B run on one of seven instruction sets scored below the base model's 18-task mean of 49.32 (range 45.29 to 48.79).
3. **Selection on noise.** A choice made on a small or reused evaluation set can reflect item sampling, seed variance, or overfitting to that benchmark's format rather than a change in ability.

[[ch-30]] defined the SFT design axes and [[ch-30a]] defined forgetting measurement. This lab runs both on a controlled set of configurations and produces a decision memo whose claims carry intervals.

## §1 Lab question, hypotheses, deliverables, and budget paths

**Question.** Does an SFT run with verified masking improve its target while passing a forgetting report on a held-out split?

**Hypotheses, written before any run.** Each row states the direction a source measured, the setting of that measurement, and the observation that would contradict it in this lab.

| Axis | Predicted direction | Source and setting | Contradicting observation |
|---|---|---|---|
| Narrow single-domain data vs diverse mix (equal size) | narrow: higher or equal target score, lower non-target scores | [[tulu-1-how-far-can-camels-go]] Table 3 (LLaMA 13B, not size-matched); [[lora-learns-less-forgets-less]] §4.2 (Llama-2-7B code IFT) | narrow arm's paired interval on forgetting suites includes zero at 3 seeds |
| Epochs 1 → 2 → 3 | non-target scores decrease with more epochs | [[lora-learns-less-forgets-less]] Table S6 (full fine-tuning 0.595, 0.579, 0.512 at 1, 2, 4 epochs, code) | epoch-3 forgetting interval overlaps epoch 1's |
| Instruction modelling (IM) vs response-only (IT), 1,030 examples | IM ≥ IT on the non-target mean on both subsets; no source prediction for which subset gains more on these suites | [[loss-masking-prompt]] Table 1 (LLaMA-2-7B, 18-task mean, IM − IT from +0.49 to +2.46 on seven datasets; no seeds reported); the larger gap for short-output data in Fig. 2 was measured on AlpacaEval 1.0 | IM − IT interval on the non-target mean lies below zero on either subset |
| LoRA vs full fine-tuning | narrow arm: LoRA forgets less; diverse arm at 2 epochs: no LoRA advantage | [[lora-learns-less-forgets-less]] Tables S6, S13 | LoRA forgets more on the narrow arm |

**LoRA** (low-rank adaptation) freezes the pretrained weights and trains, for each targeted weight matrix, a pair of matrices of rank r whose product, scaled by α/r, is added to that matrix. The **forgetting average** of [[lora-learns-less-forgets-less]] is the mean accuracy on HellaSwag, WinoGrande, and ARC-Challenge (§3.3); a lower value means more forgetting. Its tables do not print the base model's value, so its numbers show change across training durations and methods, not the drop from the base model.

**Deliverables.**
1. `tests/` with the tests of §2, each with a negative control that fails when the corresponding bug is injected.
2. `runs.jsonl`: one row per run with base checkpoint, data file hash, arm, epochs, loss type, LoRA settings, seed, code commit, and step-0 loss.
3. `eval/items/<run>/<suite>.jsonl`: the per-item score of the base model and of every checkpoint.
4. `decontamination.md`: overlap counts for every arm against every suite (§3).
5. `sft-lab-memo.md`: predictions and non-inferiority margins (the largest acceptable drop per suite, §6.4; committed before training), development results, the decision list (committed before the held-out evaluation), held-out results, and a forgetting report per run in the [[ch-30a]] §6 format.

**Budget paths.**

| Item | Full path | Resource-constrained path |
|---|---|---|
| Base model | Llama-3.1-8B (base of Llama-3.1-Tulu-3-8B-SFT) | the OLMo 2 1B base used for OLMo-2-0425-1B-SFT |
| Examples per arm | 20,000 | 10,000 |
| Axes | data arm; epochs 1/2/3; IM vs IT; LoRA vs full | data arm × epochs {1, 3} |
| Seeds per configuration | 3 | 3 |
| Unit tests, decontamination, forgetting report, held-out split | required | required |

**Compute arithmetic (derived).** Tülu 3 trained its final 8B SFT model on 939,344 prompts for 2 epochs on 32 GPUs for 6 hours, 192 GPU-hours ([[tulu-3-lab-evidence]], §4.3). Scaling by examples, a 20,000-example, 2-epoch run takes 192 × 20,000/939,344 = 4.09 GPU-hours, under the assumption that the subsample has the same mean tokens per example and throughput. The full-path matrix in §4.4 totals 76 GPU-hours of training under that assumption; evaluation time is not estimated. For the 1B path, the verified reference run (OLMo-2-0425-1B-SFT, 2 epochs over `tulu-3-sft-olmo-2-mixture-0225`) took "approximately" 9 hours on 8 H100 GPUs ([[open-instruct-allenai-recipes]], `docs/olmo2.md` L10-13); that document does not give the mixture's example count, so no per-example rate is derived.

## §2 Gate 1: unit tests before the first optimizer step

**Definition.** The gate is a test suite that checks the rendered token ids, the label vector, and the packed attention pattern against independent computations. Training starts only after it passes at the training commit.

**Problem.** Template choice alone moved Tülu 3's intermediate SFT average between 51.6 and 53.0 across five templates on Llama 3.0 ([[tulu-3-lab-evidence]], Table 13; the number of runs per template is not reported). A template or mask error changes which token ids mark turn boundaries or which positions are trained, and the training loop does not report either change.

### 2.1 Chat-template rendering and the assistant mask

**Mechanism.** In transformers v4.46.0, `apply_chat_template(..., return_assistant_tokens_mask=True, return_dict=True)` starts every mask as zeros and sets positions to 1 only inside character spans recorded from `{% generation %}` blocks. If the template has no such block, the code logs "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword." and continues ([[hf-chat-template-assistant-mask]], L1805-1808, L1895-1910). The `allenai/OLMo-2-0425-1B-Instruct` template (fetched 2026-09-15) has no such block. It renders a non-final assistant turn as `'<|assistant|>\n' + content + eos_token + '\n'` and the final one without the trailing newline.

The tests below find assistant spans by rendering prefixes, so they do not depend on `{% generation %}` (course code, not taken from a source):

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

**Negative controls.** (1) Replace the serving renderer with one that drops the trailing newline after `eos_token`; `test_training_ids_equal_serving_ids` must fail. (2) Use a training template that omits `eos_token` after one assistant turn; `test_span_is_content_plus_eos` must fail.

### 2.2 Label positions for response-only loss and instruction modelling

**Definition.** Shi et al. define instruction tuning (IT) as loss on completion tokens only and instruction modelling (IM) as loss on instruction and completion tokens excluding template tokens ([[loss-masking-prompt]] §3):

```
IT:  L = − Σ_{j=1..n} log P(C_j | I_1..I_m, C_1..C_{j−1})                    (Eq. 2)
IM:  L = − Σ_{t=1..m+n} log P(x_t | x_1..x_{t−1}) · 1(x_t ∉ T)               (Eq. 4)
```

I is the instruction of m tokens, C the completion of n tokens, x their concatenation, and T the set of template tokens such as `<|user|>`; 1(·) is 1 for non-template tokens.

**Tests.** (a) Under IT, the set of positions with a label other than −100 equals the union of the spans from §2.1 for every assistant turn. (b) Under IM, that set is a superset of the IT set and contains no position inside a role header. (c) In a batch, the count of trained tokens logged by the training loop equals the count computed by (a) or (b) from the same rows.

**Worked example.** Shi et al. Table 5 gives average lengths (unit not stated). For LIMA (total 484.47, output 442.75), IT trains 442.75/484.47 = 91.4% of positions and IM at most 100%. For Less MMLU Chat (total 225.19, output 8.24), IT trains 3.7% and IM up to 225.19/8.24 = 27.3 times as many positions. A prompt leak into the loss therefore multiplies the logged trained-token count by up to 27.3 on the second dataset but by at most 484.47/442.75 = 1.094 on the first. Test (c) compares exact counts, so it detects both; a visual check of the logged count shows the 27.3-fold change but not the 9.4% change.

### 2.3 Packed rows: labels at boundaries and cross-example attention

**Definition.** A packed row concatenates several examples. Correct packing restarts `position_ids` at 0 for each example, restricts attention to the example's own tokens, and never trains the last token of one example to predict the first token of the next.

**Mechanism** ([[packing-position-ids]], Kundu et al. §3.3-3.5).
1. The collator concatenates the examples of a mini-batch into one `input_ids` row.
2. It converts the first label of each example to −100.
3. It concatenates per-example `position_ids`.
4. With `attention_mask=None`, the attention code computes `cu_seq_len` from `position_ids` and calls `flash_attn_varlen_func()`.

**Worked example** (Kundu et al. §3.5). Four examples of lengths 4, 8, 5, 11 give `position_ids` [0,1,2,3, 0,…,7, 0,…,4, 0,…,10]. The positions where the value is 0 are 0, 4, 12, 17, so `cu_seq_len` = [0, 4, 12, 17, 28]. Of 28 labels, 4 are −100, leaving 24 trained positions before any prompt masking.

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

**Negative control.** Call the model on the packed row without `position_ids`; the second example's log-probabilities must then differ from its separate pass by more than `tol`. **Tolerance.** Set `tol` from a calibration: run the separate pass of one example twice inside two different batch compositions and take a multiple of the observed maximum difference. Record the value; no source gives a threshold.

### 2.4 Run-level checks, and three checks that do not work

1. **Step-0 loss.** Compute the base model's mean negative log-likelihood on the labeled tokens of the first training batch with a separate evaluation script. The training loop's step-0 loss must match it within a tolerance recorded in the memo. A match confirms that the same weights, labels, and loss aggregation are used.
2. **Why ln|V| is not the reference.** Karpathy's check "verify loss @ init" expects −log(1/n_classes) when "you initialize your final layer correctly" ([[karpathy-training-neural-net-recipe]], section 2). That value is the loss of a uniform prediction. For a vocabulary of 100,000 tokens it is ln 100,000 = 11.51 nats (derived). A pretrained model predicts text better than uniform, so a correct SFT run starts below this value; a step-0 loss near ln|V| indicates weights that were not loaded or an output layer that was re-initialized (Interpretation).
3. **Overfit two examples.** Train on two examples until the loss stops decreasing and confirm that the loss on labeled tokens falls toward zero and that greedy decoding reproduces the two completions ([[karpathy-training-neural-net-recipe]], "overfit one batch"). This checks the label shift and the optimizer update together.
4. **Decode the batch the model receives.** Print one packed row with labels rendered as text and −100 positions shown as blanks, following the same section's advice to decode "exactly what goes into your network".
5. **Not a test: matching packed and unpacked loss curves.** With correct position IDs, Mistral-7B on a 20K FLAN subset reached validation loss 1.129 unpacked (19,961 rows) and 1.284 with offline packing (585 rows); the authors attribute the gap to the smaller number of optimizer steps in one epoch when 19,961 examples are packed into 585 rows (§4.1). Mini-batch packing that kept the row count reached 1.127 ([[packing-position-ids]], Table 2). Curves differ for correct implementations, so equivalence is tested with the forward pass in §2.3.
6. **Not a test: a gradient-norm threshold for prompt leakage.** No source in the library gives a threshold, and a norm spike has other causes. Leakage is tested directly by §2.2 (a)-(c). Log the pre-clipping norm for diagnosis only.
7. **Loss aggregation is fixed across runs.** Tülu 3 showed that with a mean loss, gradient accumulation weights each micro-batch equally, so a token's weight depends on how many tokens share its micro-batch (Eqs. 1-2), and it switched to a sum loss ([[tulu-3-lab-evidence]], §4.3.2). Packing changes the tokens per micro-batch in the same way. The lab uses one aggregation for all arms and records it.

## §3 Data arms: a narrow single-domain set and a diverse mix of equal size

**Definition.** The **narrow arm** contains examples from one skill domain. The **diverse arm** samples every category of a general mixture in proportion to its size. Both arms have the same number of examples.

**Problem.** The published comparison used here does not match size. In [[tulu-1-how-far-can-camels-go]] Table 3, the code set had 20,022 examples and the mixture 490,694 (sum of the seven datasets' counts in Table 1), so the mixture's advantage mixes composition with quantity. The lab removes the quantity difference.

**Evidence for the direction.**
- LLaMA 13B, Code-Alpaca vs base: Codex-Eval 28.6 → 34.2, MMLU 42.3 → 42.5, GSM 14.5 → 13.5, BBH 39.3 → 35.6, TyDiQA 43.2 → 38.9. Human+GPT mixture: 49.3, 40.5, 43.3, 45.6 on MMLU, GSM, BBH, TyDiQA and 35.9 on Codex-Eval ([[tulu-1-how-far-can-camels-go]] Table 3). GSM uses 200 of 1,319 test items (App. E), so the −1.0 GSM change is inside the analytic half-width of 4.9 points at 14.5% (derived). Result (single study).
- LLaMA-2-7B response-only SFT on single sources of 1,030-13,533 examples: 18-task mean 45.29-48.79 against 49.32 for the base; Alpagasus Alpaca 5k lowered Commonsense Reasoning from 75.86 to 66.06 ([[loss-masking-prompt]] Table 1). Result (single study).
- Llama-2-7B full fine-tuning: "programming induces more forgetting than math"; forgetting average after 2 epochs 0.579 on Magicoder-Evol-Instruct-110K and 0.599 on MetaMathQA, and on the code set it fell to 0.414 at 16 epochs ([[lora-learns-less-forgets-less]] §4.2, Tables S6, S8). Result (single study). Tülu 1 measures the drop from the base model on BBH and TyDiQA; Biderman et al. measure a decrease with training duration without a printed base value. Both point to lower non-code scores after code instruction data, in different settings and metrics.
- Tülu 3 8B: removing math data lowered MATH 31.5 → 23.5 and the unseen DeepMind Mathematics 32.3 → 23.3, so that skill-specific data transferred to an unseen benchmark ([[tulu-3-lab-evidence]], Table 32).

**Construction.**
1. **Diverse arm (full path).** Sample 20,000 examples from the Tülu 3 SFT mixture stratified by the Table 7 categories ([[tulu-3-sft-mix]]; category sums in [[tulu-3-lab-evidence]]). The resulting counts are in the table below.
2. **Narrow arm.** Sample 20,000 examples from Tülu 3 Persona Python (34,999 examples; GPT-4o problems conditioned on personas and claude-3-5-sonnet programs, [[tulu-3-lab-evidence]] §3.1.2). The learner's [[ch-29]] pool restricted to one category is an alternative when it has 20,000 rows.
3. **Token accounting.** Record trained tokens and optimizer steps per arm. When the arms differ in trained tokens, report both and add one token-matched diverse run.
4. **Decontamination.** Apply the Tülu 3 rule to both arms against every development, held-out, and forgetting suite (see worked example).
5. **Representation labels.** For each suite in §5, record whether the arm contains data for that skill.

**Worked example: stratified counts** (derived from Table 7 category sums, largest-remainder rounding).

| Category | SFT prompts | Share | Examples in a 20,000-example diverse arm |
|---|---|---|---|
| Math reasoning | 334,252 | 35.58% | 7,117 |
| Coding | 142,275 | 15.15% | 3,029 |
| General | 116,872 | 12.44% | 2,489 |
| Safety and non-compliance | 110,983 | 11.81% | 2,363 |
| Knowledge recall | 104,982 | 11.18% | 2,235 |
| Multilingual | 100,000 | 10.65% | 2,129 |
| Precise IF | 29,980 | 3.19% | 638 |
| Total | 939,344 | 100% | 20,000 |

**Worked example: decontamination rule** ([[tulu-3-lab-evidence]], §3.2). A GSM8K test question has 40 tokens. If 22 of its tokens lie inside 8-grams shared with one training example, 22/40 = 55% > 50%, so the question overlaps that example. GSM8K has 1,319 test questions; 2% of 1,319 is 26.38, so a training source that overlaps 27 or more questions is contaminated for GSM8K. For a held-out suite the source is removed; for a development suite the whole source is removed if that does not significantly affect performance; otherwise only the matching examples are removed.

**Implication for a general-purpose model.** The narrow arm measures what a single-domain synthetic set costs outside its domain at fixed size. The diverse arm is the control that a general-purpose recipe would use.

## §4 Training axes that change generality

### 4.1 Epochs 1, 2, 3

**Problem.** Forgetting grew with training duration in [[lora-learns-less-forgets-less]] (§4.2): full fine-tuning on code had forgetting averages 0.595, 0.579, 0.512, 0.446, 0.414 at 1, 2, 4, 8, 16 epochs, while HumanEval peaked at 0.497 at 8 epochs (Tables S5-S6). On the diverse Tülu-v2-mix, full fine-tuning's forgetting average was 0.660, 0.652, 0.621 at 2, 4, 6 epochs (Table S13). Tülu 3 found that training longer than 2 epochs did not improve its average on the Tülu 2 mixture (§4.3.2, curves without numeric labels).

**Mechanism: separate runs, not checkpoints of one run.** The code IFT runs above were "separate models for 1, 2, 4, 8, and 16 epochs" (§4.1). The Tülu-v2-mix numbers come from "hot" checkpoints of one 6-epoch run without per-duration cooldown (App. C). The two designs are not equivalent.

**Worked example (derived).** Tülu 3's schedule is linear decay after a 3% warmup ([[tulu-3-lab-evidence]], Table 11). In a 3-epoch run of T steps, the learning rate at step T/3 is (1 − 1/3)/(1 − 0.03) = 0.687 of the peak, 3.44×10⁻⁶ for a 5×10⁻⁶ peak. A 1-epoch run with the same schedule ends at 0. An epoch-1 checkpoint of the 3-epoch run is therefore taken while the learning rate is still 69% of its peak. The lab trains separate runs for 1, 2, and 3 epochs.

### 4.2 Instruction modelling versus response-only loss on 1,030-example subsets

**Evidence.** On LIMA (1,030 examples; instruction/output length ratio 0.0942), IM changed the 18-task mean from 48.79 to 49.60, MT-Bench from 4.77 to 4.83, and AlpacaEval 1.0 from 33.06 to 32.94 ([[loss-masking-prompt]] Table 1, Table 5). On AlpacaEval 1.0, the IM advantage was larger for datasets with long instructions and short outputs, such as Less MMLU Chat (ratio 26.3; 4.42 → 9.78), and grew as Tülu V2 subsets shrank toward 1,000 examples at a ratio near 10 (Fig. 2; both panels plot AlpacaEval 1.0 improvement). On the 18-task mean, Table 1 shows no such ordering: IM − IT was +0.81 for LIMA and +0.66 for Less MMLU Chat, and the three largest gains (+2.18 to +2.46) were on Alpagasus sets with ratios 0.30-0.64 (Tables 1 and 5). Mechanism evidence: on LIMA, IM had higher training loss on outputs (1.45 vs 1.37) and lower test loss on Tülu V2 (1.17 vs 1.32) (Fig. 3). No seeds or intervals are reported. Result (single study).

**Design.** Two subsets of 1,030 examples: subset L with long outputs (LIMA, or diverse-arm examples with instruction/output ratio below 0.1) and subset S with short outputs (ratio at least 10, the fixed ratio of Fig. 2 right). Each is trained with IT and IM for 2 epochs, 3 seeds. For the lab's non-target suites, the source supports IM ≥ IT on both subsets. Whether the difference is larger on subset S for these suites is an Open question: the ratio dependence in the source was measured on AlpacaEval 1.0, which the lab does not run. The loss type is the only change: LR, batch, and epochs follow the lab row for the model size.

**Limit.** Shi et al. write "we are not proposing IM as a replacement for current fine-tuning processes" (Abstract). The axis tests the overfitting mechanism on small sets; it does not choose the loss for the 20,000-example arms.

### 4.3 LoRA versus full fine-tuning

**Evidence.** Narrow code data, Llama-2-7B, epoch 4: full fine-tuning HumanEval 0.470 with forgetting average 0.512; LoRA r = 256 0.498 with 0.631; LoRA r = 16 0.358 with 0.652 ([[lora-learns-less-forgets-less]] Tables S5-S6). At similar target accuracy (r = 256 vs full fine-tuning at epoch 4), LoRA scored 2.8 points higher on HumanEval and 11.9 points higher on the forgetting average (derived). Diverse Tülu-v2-mix, epoch 2: forgetting average 0.660 for full fine-tuning and 0.650 for LoRA r = 16, and "all LoRA models are within one standard error" of full fine-tuning on MT-Bench; at epoch 6 full fine-tuning forgot the most (Tables S10-S13, App. C.2). Each condition was trained once (§4.1). Result (single study).

**Lab settings.** LoRA on the seven projection matrices of every block (W_q, W_k, W_v, W_o, W_gate, W_up, W_down, as in App. A) with r = 256 and α = 512, the only rank whose HumanEval reached full fine-tuning's in Table S5 (0.498 vs 0.470 at epoch 4). The LoRA learning rate is chosen on the development suite from {1×10⁻⁴, 2×10⁻⁴}, the values used for r = 256 and r = 16/64 in App. A; App. B found LoRA's best learning rate an order of magnitude above full fine-tuning's.

### 4.4 Run matrix (full path)

| Configuration | Arm | Epochs | Loss | Adapter | Seeds | GPU-hours (derived, §1) |
|---|---|---|---|---|---|---|
| F1 baseline | diverse | 2 | IT | full | 3 | 12.3 |
| F2 | narrow | 2 | IT | full | 3 | 12.3 |
| F3, F4 | diverse | 1; 3 | IT | full | 3 + 3 | 6.1 + 18.4 |
| F5, F6 | diverse; narrow | 2 | IT | LoRA r = 256 | 3 + 3 | 12.3 + 12.3 |
| F7 | subsets L and S (1,030 each) | 2 | IT; IM | full | 12 runs | 2.5 |

The total is 30 runs and 76 GPU-hours of training under the equal-throughput assumption. The assumption is weakest for F5 and F6: [[lora-learns-less-forgets-less]] reports that "for standard implementations and a fixed batch size, LoRA tends to train slower than full finetuning" (§4.7, App. I). The constrained path runs {narrow, diverse} × {1, 3} epochs with full fine-tuning and IT: 4 configurations, 12 runs at 10,000 examples.

## §5 Evaluation split: development suite, held-out suite, and forgetting suites

### 5.1 Why development and held-out suites are separate benchmarks

**Definition.** The **development suite** is used for every choice: LoRA learning rate, run acceptance, and the interpretation of each axis. The **held-out suite** contains a different benchmark for each skill and is evaluated once, after the decision list is committed. Tülu 3 used this design and "did not examine scores on our unseen set when developing our models" ([[tulu-3-lab-evidence]], §2.2, Table 3).

**Evidence.** In Tülu 3's SFT ablations, IFEval dropped from 72.8 to 53.6 without the Persona datasets (the removal covers the Persona math, coding, and Precise IF sets, §4.2 and Table 10) while IFEval-OOD did not drop (17.6 vs 18.0). The authors conclude that their choices "overfit to the development evaluations in Precise Instruction Following" (§7.4.1). Persona IF was generated to cover the "25 different constraint types defined in IFEval" (§3.1.2). In a second Ai2 study, many models score above 80% on IFEval's 25 constraint templates, while GPT-4.1 and Claude 3.7 Sonnet score below 50% on IFBench's 58 unseen constraints ([[ifbench]] §1, Fig. 1). The two results agree that IFEval scores overstate constraint following outside its 25 types; they come from overlapping author groups, so each is a Result (single study) rather than an independent replication.

A random split of one benchmark's items into development and held-out halves removes selection on item noise but not overfitting to that benchmark's format. The lab uses separate benchmarks (Interpretation of the Tülu 3 design).

### 5.2 Suite roles

| Suite | Skill | Items | Role | Narrow (code) arm | Diverse arm |
|---|---|---|---|---|---|
| MMLU | knowledge | 14,042 ([[benchmark-variance-quantified]] Table 1) | forgetting report; development | not represented | knowledge-recall category present |
| ARC-Challenge | science reasoning | 1,165 (as evaluated in the same table) | forgetting report | not represented | no dedicated category |
| GSM8K | grade-school math | 1,319 (same table) | forgetting report; development | not represented | represented (math reasoning) |
| TriviaQA | closed-book factual recall | 11,313 (same table) | forgetting report | not represented | no dedicated category |
| IFEval | precise instruction following | 541 prompts ([[ifeval]] §1) | forgetting report; development | not represented | represented (Persona IF uses its 25 types) |
| HumanEval | code | 164 (same table) | target; development | represented | represented |
| MMLU-Pro, AGIEval English, DeepMind Mathematics, BigCodeBench-Hard (148 of 1,140), IFEval-OOD (52 constraints) or IFBench (58 constraints, 300 prompts) | one per skill | as listed | held-out | — | — |

A decrease on a suite marked "not represented" is forgetting. A decrease on a represented suite is a failure to reach the target. For the diverse arm, GSM8K and IFEval measure targets, and the held-out partners DeepMind Mathematics and IFEval-OOD or IFBench measure whether those targets transfer.

### 5.3 Protocol

1. Fix prompt templates, shot counts, decoding, and answer extraction per suite, and use them for the base model and every checkpoint. Report IFEval prompt-level loose accuracy, as in Tülu 3 (§7.2), and prompt-level strict accuracy, because the loose criterion can accept a response that fails a word count after its first line is removed ([[ifeval]] §2.2).
2. For suites that a base model cannot follow as a chat task, use one completion-style formulation for base and SFT checkpoints. Tülu 1 did not evaluate vanilla LLaMA on AlpacaEval "due to them having little instruction-following ability without further prompt engineering" ([[tulu-1-how-far-can-camels-go]] §4.2). A base-versus-SFT IFEval difference mixes format with ability, so the IFEval entry compares arms at the same chat format (narrow vs diverse), not base vs SFT (Interpretation). Because the diverse arm contains 638 Precise IF examples built from IFEval's constraint types (§3), this entry is reported and labeled as an arm difference, not as forgetting.
3. When a base model scores near chance on standard multiple-choice MMLU, use the cloze formulation: for the 7B seed models of [[benchmark-variance-quantified]], standard MMLU stayed at 25.86 with monotonicity 0.09 and MMLU-Cloze reached 37.47 with monotonicity 0.95 (Table 1, §3.3). Monotonicity is the Kendall rank correlation between a seed model's scores across its training checkpoints and an increasing sequence for discrete metrics or a decreasing sequence for continuous metrics (§3.1).
4. Store per-item scores. Every interval in §6 is computed from them.
5. Commit the decision list before running the held-out suite, and report held-out results for every configuration.

## §6 Statistics: paired intervals, bootstrap, and seeds

### 6.1 Item-level interval for one comparison

**Formula.** For a 0/1 score with accuracy S on N items, the analytic 95% half-width is 1.96 · sqrt(S(1 − S)/N) ([[benchmark-variance-quantified]] §3.1). For two checkpoints A and B scored on the same items ([[adding-error-bars-evals]] §4.1-4.2):

```
SE_unpaired = sqrt(SE_A² + SE_B²)
SE_paired   = sqrt( (1/(n−1)) Σ_i (d_i − d̄)² / n )          (Eq. 7)
CI_95%      = d̄ ± 1.96 × SE                                  (Eq. 5)
```

SE_A and SE_B are the standard errors of each checkpoint's mean score (for a 0/1 score, sqrt(S(1 − S)/n)); d_i is checkpoint A's score minus checkpoint B's score on item i; d̄ is the mean of d_i over the n items; SE is SE_unpaired or SE_paired.

**Worked example: suite size.** At 60% accuracy, IFEval's 541 prompts give a half-width of 1.96 · sqrt(0.6 × 0.4/541) = 4.1 points, ARC-Challenge's 1,165 items at 40% give 2.8 points, and HumanEval's 164 problems at 30% give 7.0 points when each problem is scored 0/1 (derived; pass@k averaged over samples is not a 0/1 score, and the formula then serves only as an approximation). MT-Bench has 80 questions, 10 in each of 8 categories ([[judge-llm-bias]] §2.2); on Tülu-v2-mix all LoRA runs were within one standard error of full fine-tuning on it ([[lora-learns-less-forgets-less]] App. C.2). The lab does not use 10-question category slices for decisions.

**Worked example: paired forgetting delta.** The base model answers 660 of 1,319 GSM8K questions (50.0%). After narrow-arm SFT, 110 of those become wrong and 70 wrong ones become right, so the SFT checkpoint scores 620 (47.0%).
- d̄ = (70 − 110)/1,319 = −0.0303.
- Σ d_i² = 180, so Σ(d_i − d̄)² = 180 − 1,319 × 0.0303² = 178.79, the sample variance is 178.79/1,318 = 0.1357, and SE_paired = sqrt(0.1357/1,319) = 0.0101.
- Paired CI: −0.0303 ± 0.0199 = [−0.050, −0.010]; it excludes zero.
- SE_unpaired = sqrt(0.500 × 0.500/1,319 + 0.470 × 0.530/1,319) = 0.0195; CI [−0.068, +0.008]; it includes zero.
- With the same net change from 220 losses and 180 gains, SE_paired = 0.0151 and the paired CI is [−0.060, −0.001].

The figure [figures/forgetting-ci-planner.html](figures/forgetting-ci-planner.html) lets the reader pick a suite size, set the base score, loss and gain counts, seed count, seed standard deviations, and number of comparisons, and see the unpaired, paired, seed-inclusive, and Bonferroni-adjusted intervals against zero.

### 6.2 Paired bootstrap

**Definition.** A paired bootstrap resamples item indices with replacement, recomputes the mean difference on each resample, and takes the 2.5th and 97.5th percentiles of the resampled means as the 95% interval. It does not use the normal approximation behind the 1.96 multiplier, and it applies to non-binary scores such as F1 or judge ratings.

```python
import numpy as np   # course code, not taken from a source

def paired_bootstrap_ci(ref, new, n_boot=10_000, alpha=0.05, seed=0, chunk=500):
    d = np.asarray(new, float) - np.asarray(ref, float)      # same items, same order
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), (min(chunk, n_boot - b), len(d)))].mean(1)
                            for b in range(0, n_boot, chunk)])  # chunks bound memory for 14,042-item suites
    return d.mean(), np.quantile(means, [alpha / 2, 1 - alpha / 2])
```

**Worked example (checkable by hand).** Five items have differences d = [1, 0, 0, −1, 1], so d̄ = 0.2 and Σ(d_i − d̄)² = 0.64 + 0.04 + 0.04 + 1.44 + 0.64 = 2.8. Over all 5⁵ = 3,125 equally likely resamples, the variance of the resampled mean is (2.8/5)/5 = 0.112, a standard error of 0.335, and the percentile interval is [−0.4, 0.8] (enumerated). Eq. 7 gives sqrt((2.8/4)/5) = 0.374. The bootstrap standard error is smaller by the factor sqrt((n − 1)/n) = 0.894, which matters only for small n.

### 6.3 Seed variance

**Evidence.** Tülu 3 8B SFT seed averages were 59.9, 60.1, 59.8, 59.8, 59.8 (standard deviation 0.13, derived) and 70B averages 71.8, 70.0, 72.6 (standard deviation 1.33, derived); the authors call the variation a reason for "multiple training runs" ([[tulu-3-lab-evidence]], Table 14). For ten 7B models pretrained with different seeds, the seed standard deviation was below the item-level 95% half-width on every benchmark in Table 1, for example ARC-C 0.80 vs 2.74 and MMLU 0.57 vs 0.72 ([[benchmark-variance-quantified]] Table 1). Those are pretraining seeds; no source here measures SFT seed variance per benchmark. Each measurement is a Result (single study) for its stage.

**Formula (course approximation, derived, not from a source).** For arms A and B with k_A and k_B seeds, per-seed mean scores with sample standard deviations s_A and s_B, and a paired item-level standard error SE_items computed on the seed-averaged per-item scores:

```
SE_total ≈ sqrt( SE_items² + s_A²/k_A + s_B²/k_B )
```

For a base-versus-SFT comparison the base term s_B is 0. The approximation treats training randomness on a fixed item set and item sampling as independent.

**Worked example (illustrative scores).** Narrow arm GSM8K by seed: 47.0, 45.8, 48.1 (mean 46.97, s_A = 1.15). Diverse arm: 49.5, 50.2, 48.9 (mean 49.53, s_B = 0.65). Difference −2.57 points. With SE_items = 1.0 point, the seed terms add 1.15²/3 + 0.65²/3 = 0.58, SE_total = sqrt(1.00 + 0.58) = 1.26, and the 95% interval is [−5.03, −0.10]. With one seed per arm the seed terms cannot be estimated and the interval would use SE_items alone, [−4.53, −0.61], which is narrower than the data support.

**Two-level bootstrap.** Resample seeds within each arm and items jointly:

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

**Limit.** With 3 seeds there are 10 distinct seed multisets per arm, so the seed part of the bootstrap distribution takes at most 10 values per arm. Report the per-seed scores next to every interval.

### 6.4 Decision rule and multiple comparisons

1. Before training, write for each forgetting suite a non-inferiority margin δ (the largest acceptable drop) and for each target suite the minimum gain of interest ([[ch-30a]] §6, item 12).
2. A run passes the forgetting report on a suite when the lower end of its seed-inclusive 95% interval for SFT − base is above −δ.
3. An axis effect is reported as detected when the interval for the difference between configurations excludes zero after correction.
4. **Bonferroni correction.** With m comparisons, test each at α/m. For m = 20 (5 forgetting suites × 4 axes), α/m = 0.0025 and the two-sided multiplier is 3.02 instead of 1.96 (derived). The worked seed example becomes −2.57 ± 3.02 × 1.26 = [−6.37, +1.24], which includes zero. [[ch-51]] treats interval-based go/no-go decisions in more depth.
5. When a benchmark's scores are too noisy to decide at the lab's scale, prefer a lower-noise formulation or metric over adding more comparisons: in pretraining ablations at 60M-1B, a 1,000-question ARC-Easy subset gave higher decision accuracy than MMLU with 90% fewer instances ([[signal-and-noise-eval]] App. B.2). Result (single study; not tested for SFT).

## §7 Acceptance criteria and memo

A run and the lab are accepted when all of the following hold.
1. `pytest tests/` passes at the training commit, and each negative control in §2 fails when its bug is injected. Both outputs are committed.
2. Each run's step-0 loss equals the independent base-model loss on the same labeled tokens within the recorded tolerance (§2.4 item 1), and the two-example overfit check reproduces both completions.
3. `runs.jsonl` has 3 seeds for every configuration, with data hashes and commits; no run is dropped after evaluation.
4. `decontamination.md` reports the Tülu 3 overlap counts for every arm and suite, with the action taken.
5. Every run has a forgetting report in the [[ch-30a]] §6 format with paired, seed-inclusive 95% intervals on MMLU, ARC-Challenge, GSM8K, TriviaQA, and IFEval (IFEval as an arm-versus-arm comparison labeled as such, §5.3), each suite labeled represented or not represented. Items of the §6 format that the lab does not measure, such as the safety pair and the recovery test, are listed as not measured.
6. Predictions and margins were committed before the first training run, and the decision list before the held-out evaluation; `git log` shows the order.
7. The memo reports all configurations, including those whose intervals include zero, and selects no seed by its held-out score.

**Memo sections.** (1) Setup: base model, arms with example and token counts, runs, hardware. (2) Development table: per suite and configuration, mean over seeds, per-seed scores, interval. (3) Decisions made from the development table. (4) Held-out table computed once, with the development-minus-held-out gap per skill. (5) Forgetting report per run. (6) Hypotheses from §1 marked supported, contradicted, or undecided, with the interval that decided each.

## Recipe

All rows were read at the stated locus; dates are given in the Status column.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value | Lab value | Reason for difference |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | peak LR; schedule; warmup; effective batch; max length; epochs | 5×10⁻⁶; linear; 0.03; 128; 4,096; 2 | arXiv:2411.15124v5 §4.3, Table 11 ([[tulu-3-lab-evidence]]) | verified 2026-09-15 | "found after a hyperparameter search" (§4.3); §4.3.2 Figs. 5-6 on Llama 3.0 + Tülu 2 mix, no numeric labels | full path: same, epochs 1/2/3 on the epoch axis | epochs are a tested axis |
| Llama-3.1-Tulu-3-8B-SFT | 8B | SFT | loss aggregation | sum (paper); `--reduce_loss sum` at open-instruct 8781471; flag absent, mean default at 098424c | §4.3.2; `docs/tulu3.md`@8781471 L49 ([[open-instruct-allenai-recipes]]) | conflict (paper and June 2025 command vs current default) | §4.3.2 Figs. 5-6 | sum for every run | per-token weights must not change with packing or GPU count (§2.4) |
| Llama-3.1-Tulu-3-8B-SFT; -70B-SFT | 8B; 70B | SFT | seeds; checkpoint selection | 42, 123, 456, 789, 1011 (8B); 42, 123, 456 (70B); best single run released | Table 14, §4.3.1 | verified 2026-09-15 | Table 14 averages 59.8-60.1 (8B), 70.0-72.6 (70B) | 3 seeds; all reported; no selection | selection by a reported score biases that score upward (Interpretation) |
| Tülu 3 (all sizes) | — | eval-gate | decontamination | 8-gram on user turns; >50% of test tokens matched to one training instance; source contaminated if >2% of any evaluation's instances overlap | §3.2 | verified 2026-09-15 | §3.2 comparison of full-string, n-gram, and embedding matching | same rule against development, held-out, and forgetting suites | none |
| OLMo-2-0425-1B-SFT | 1B | SFT | LR; effective batch; epochs; max length; warmup; data | 3×10⁻⁵; 8 × 2 × 8 = 128; 2; 4,096; 0.03; `tulu-3-sft-olmo-2-mixture-0225` | open-instruct@098424c `docs/olmo2.md` L45-53 ([[open-instruct-allenai-recipes-recipe]]) | verified 2026-09-14 | no ablation reported | constrained path: same LR, batch, length, warmup; epochs 1 and 3; 10,000 examples per arm | epochs are the tested axis; example count set by budget |
| OLMo-2-0425-1B-SFT | 1B | SFT | loss reduction | `--reduce_loss sum` at 5e5c93d (L50); absent at 098424c | `docs/olmo2.md`@5e5c93d L50 | conflict | PR #1024 changed the default to mean | sum | as above |
| Llama-2-7B IT and IM runs (Shi et al.) | 7B | SFT | LR; total batch; epochs; max length; optimizer; schedule; weight decay | 2×10⁻⁵; 128; 2, 3, or 10 ("typically" 2); 2,048; AdamW (0.9, 0.98), ε 1e-6; linear, warmup 0.03; 0 | arXiv:2405.14394v2 App. C, Table 6 ([[loss-masking-prompt]]) | verified 2026-09-15 | no ablation of these values reported | F7 uses the lab's LR and batch for the model size, 2 epochs | the loss type is the only change within F7 |
| Llama-2-7B → Magicoder-Evol-Instruct-110K, LoRA | 7B | SFT | rank; α; LoRA LR; global batch; length; optimizer; schedule; epochs | r = 16, 64, 256; α = 2r; 2×10⁻⁴ (r = 16, 64), 1×10⁻⁴ (r = 256); 192; 4,096; decoupled LionW (0.9, 0.95); cosine, warmup 0.1; separate runs at 1, 2, 4, 8, 16 | arXiv:2405.09673v2 §4.1, App. A ([[lora-learns-less-forgets-less]]) | verified 2026-09-15 | Tables S5-S6; App. B 2-epoch LR sweep | r = 256, α = 512, all modules; LR from {1×10⁻⁴, 2×10⁻⁴} on development suite; 2 epochs | base model and optimizer differ from the source |
| Llama-2-7B → Magicoder; → MetaMathQA, full FT | 7B | SFT | best full fine-tuning LR in a 2-epoch sweep | 5×10⁻⁵ (code); 1×10⁻⁵ (math) | App. B | verified 2026-09-15 | Fig. S1 | not used | lab uses the Tülu 3 and OLMo 2 values for its base models |
| Llama-2-7B → Tülu-v2-mix | 7B | SFT | LR; global batch; checkpoints | full 5×10⁻⁶, LoRA 1×10⁻⁴; 192; evaluated at 2, 4, 6 epochs of one run without per-duration cooldown | App. C, C.1 | verified 2026-09-15 | Tables S10-S13 | separate runs per epoch count | hot checkpoints are taken at a nonzero LR (§4.1) |
| Tülu (LLaMA 7B, 13B) | 7B; 13B | SFT | epochs; LR; schedule; max length; loss mask | 2; 2×10⁻⁵; linear warmup 3%, linear decay; 2,048; assistant tokens only | arXiv:2306.04751v2 App. D, §3.2 ([[tulu-1-how-far-can-camels-go]]) | verified 2026-09-14 | no ablation reported | not used for settings | cited for the Table 3 data comparison |

**Starting point for a small general-purpose run.** For an 8B base model trained on a stratified subsample of the Tülu 3 SFT mixture, use peak LR 5×10⁻⁶ with linear decay and 3% warmup, effective batch 128, maximum length 4,096, 2 epochs, and a sum loss; Tülu 3 used these values for Llama 3.1 8B on all 939,344 prompts on 32 GPUs, and they were not re-tuned for 20,000 examples. For a 1B base model, use the OLMo-2-0425-1B-SFT values: LR 3×10⁻⁵, effective batch 128, maximum length 4,096, warmup 0.03, 2 epochs, on the `tulu-3-sft-olmo-2-mixture-0225` data with 8 H100 GPUs. For LoRA, r = 256 with α = 512 on all modules and LR 1×10⁻⁴ is the setting used for Llama-2-7B on 110K code examples and on Tülu-v2-mix with LionW; with AdamW the LR is not verified and is chosen on the development suite.

## Generalization lens

**(a) What increases breadth.**
- Mixing skill categories: the Tülu 1 mixture scored higher than the code-only set on every evaluation at 13B, without size matching ([[tulu-1-how-far-can-camels-go]] Table 3); the lab's equal-size arms test the composition part of that result.
- Skill data whose gain transfers to an unseen benchmark: Tülu 3 math data raised MATH by 8.0 points and the unseen DeepMind Mathematics by 9.0 points ([[tulu-3-lab-evidence]], Table 32).
- Limiting the size of the update on narrow data: at epoch 4, LoRA r = 256 reached HumanEval 0.498 against 0.470 for full fine-tuning, with a forgetting average of 0.631 against 0.512 ([[lora-learns-less-forgets-less]] Tables S5-S6).
- Loss on instruction tokens for small instruction sets: IM raised the 18-task mean over IT on all seven datasets in Table 1 of [[loss-masking-prompt]] (+0.49 to +2.46; no seeds reported), and its AlpacaEval 1.0 advantage was larger for short-output data and fewer examples (Fig. 2).

**(b) What causes narrowing or forgetting.**
- Single-domain SFT data: code-only data lowered BBH and TyDiQA at 13B ([[tulu-1-how-far-can-camels-go]]); every single-source IT run in [[loss-masking-prompt]] Table 1 scored below the base 18-task mean.
- More epochs: forgetting average fell with epochs for full fine-tuning on code (separate runs) and from epoch 2 to epoch 6 on Tülu-v2-mix (0.660, 0.652, 0.621; checkpoints of one run) ([[lora-learns-less-forgets-less]] Tables S6, S13).
- Training data built from a benchmark's constraint list: the Persona datasets, which include a Precise IF set built from IFEval's 25 constraint types, moved IFEval by 19.2 points and IFEval-OOD by −0.4 ([[tulu-3-lab-evidence]], Table 32); many models score above 80% on IFEval while GPT-4.1 and Claude 3.7 Sonnet score below 50% on IFBench ([[ifbench]] §1).
- More SFT data on one metric: in Tülu 3's stratified subsamples, "TruthfulQA performance actually drops as the amount of data in the mix increases" (§4.2, Fig. 4, no numeric labels).

**(c) How to measure it at this stage.**
- Paired base-versus-SFT differences per suite, labeled represented or not represented, with seed-inclusive intervals (§6; [[adding-error-bars-evals]] §4.2).
- The development-minus-held-out gap per skill, using different benchmarks per skill ([[tulu-3-lab-evidence]], Tables 3 and 32).
- Suite sizes large enough for the effect of interest: a 164-item code benchmark at 30% has a 7.0-point half-width, and an 80-question chat benchmark did not separate LoRA from full fine-tuning ([[lora-learns-less-forgets-less]] App. C.2).
- Formulations with measurable signal at the model's scale, such as cloze MMLU when standard MMLU is near chance (25.86 for the 7B seed models after 210B tokens; [[benchmark-variance-quantified]] Table 1, §3.3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing a pretrained model's step-0 loss with ln\|V\| | every correct run "fails" an initial-loss gate | compare with the base model's independently computed loss on the same labeled tokens (§2.4) |
| Requiring packed and unpacked training curves to match | correct packing is rejected; validation losses differ with step count (1.284 vs 1.129 in [[packing-position-ids]] Table 2) | forward-pass log-probability equality test (§2.3) |
| Using a gradient-norm threshold as a prompt-leakage test | false alarms from unrelated spikes; leakage without spikes | trained-token count test (§2.2 c) |
| `return_assistant_tokens_mask=True` with a template lacking `{% generation %}` | warning in logs; `assistant_masks` all zero | `test_hf_mask_not_empty_if_used`; search the template for the keyword |
| First label of each packed example not masked | trained-token count exceeds expected by one per example | boundary label test (§2.3) |
| Treating IFEval as held-out when the mix contains IFEval-constraint data | IFEval gain without an IFEval-OOD or IFBench gain | suite-role table (§5.2); held-out partner per skill |
| Arms differ in example or token count | narrow-vs-diverse difference changes when re-run at matched tokens | report trained tokens and steps per arm (§3) |
| Reporting an epoch-1 checkpoint of a 3-epoch run as a 1-epoch run | epoch-1 scores differ from a separately trained 1-epoch run | `runs.jsonl` records schedule horizon and stop step (§4.1) |
| One seed per configuration | a difference smaller than the seed spread is reported as an effect | 3 seeds; seed-inclusive interval (§6.3) |
| Deciding on MT-Bench categories of 10 questions | per-category deltas smaller than their item-level half-width are reported as effects | half-width computed before use (§6.1) |
| Choosing the best seed or epoch on the held-out suite | held-out score above development trend; no committed decision list | commit order in `git log` (§7 item 6) |
| Mean loss aggregation with different packing or GPU counts across arms | same data gives different results at different batch layouts | fixed sum aggregation recorded in `runs.jsonl` (§2.4 item 7) |
| Interpreting base-vs-SFT IFEval change as forgetting | base model fails the chat format regardless of ability | compare arms at the same format (§5.3) |

## Check your understanding

1. The narrow-arm run improves HumanEval by 5 points and its GSM8K paired interval is [−4.1, −0.3]. The diverse-arm run's GSM8K interval is [−1.0, +1.5]. Explain which result is forgetting and why the diverse arm's GSM8K entry is not a forgetting measurement.
2. A 3-epoch run's epoch-1 checkpoint forgets less than a separate 1-epoch run. Using the learning-rate arithmetic in §4.1, explain two reasons the two checkpoints differ, and state which design supports a claim about epoch count.
3. Explain why a packed run with correct position IDs can have a higher validation loss after one epoch than an unpacked run, and why this does not indicate cross-example attention.
4. Removing the Persona datasets, including the Precise IF set built from IFEval's constraint types, lowered IFEval by 19.2 points but did not lower IFEval-OOD. Give the causal account the Tülu 3 authors offer and explain what the lab would conclude if its diverse arm beat the narrow arm on IFEval but not on IFBench.
5. In §6.3, the interval excludes zero before correction and includes zero after Bonferroni correction for 20 comparisons. Explain what changed, and what the memo should report about that axis.
6. Shi et al. find a larger IM advantage on AlpacaEval 1.0 for Less MMLU Chat than for LIMA, but not on the 18-task mean. Using the token-share arithmetic of §2.2, explain how the instruction/output ratio changes what fraction of the gradient comes from instruction tokens, and why that fraction need not predict the change on non-target suites.
7. On narrow code data at epoch 4, LoRA r = 256 forgot less than full fine-tuning at a similar HumanEval score (0.498 vs 0.470), but on Tülu-v2-mix at 2 epochs full fine-tuning forgot less (forgetting average 0.660 vs 0.650 for LoRA r = 16). Propose a mechanism that accounts for both results and an observation in this lab that would contradict it.
8. Explain why a step-0 loss near ln|V| is evidence of a loading error for a pretrained model, while the same value is the expected result for a randomly initialized output layer.

## Connections

- Previous chapter: [[ch-35a]] — Distillation in Practice B: Prompt Selection, Teacher Sampling, and Quality Filters.
- Next chapter: [[ch-15]] — Human Preference and Instruction Data: Annotation Protocols, Agreement, and Prompt Coverage.
- Dependency: [[ch-30a]] — Forgetting and Alignment Tax in Fine-Tuning: Measurement and Control (paired forgetting delta, report format §6).
- [[ch-04]] — Sequence Packing, Loss Masking, and Chat Templates (mechanics tested in §2).
- [[ch-30]] — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (the axes run in §4).
- [[ch-30b]] — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (mixture composition beyond the two arms).
- [[ch-29]] — Lab: Synthetic Instruction Set with Filter, Deduplication, and Verification (optional source for a narrow synthetic arm).
- [[ch-31a]] — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood (this lab trains on positive targets only; decontamination removes examples as negative marginal value).
- [[ch-47]] — Evaluation Harness and Suite Design for General Capability (depends on this lab); [[ch-51]] — Metric Noise, Confidence Intervals, and Go/No-Go Decisions; [[ch-53]] — Lab: Evaluation Harness with a Held-Out Suite, Forgetting Report, and Perturbation Robustness (reuses this lab's per-item files).
- [[ch-46]] — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out Capability Retention (applies the same forgetting report after preference or RL training).

## Sources

- [[tulu-3]] — Tülu 3 report (arXiv:2411.15124v5); library card for the report.
- [[tulu-3-lab-evidence]] — excerpt of Tülu 3 Tables 3, 7, 11, 13, 14, 32 and §3.2, §4.3.2, §7.4.1 used for the suite split, decontamination rule, seeds, and recipe rows.
- [[tulu-3-sft-mix]] — Tülu 3 SFT mixture components used to build the diverse and narrow arms.
- [[loss-masking-prompt]] — Shi et al., IT and IM objectives (Eqs. 2, 4), Tables 1, 5, 6, Figs. 2-3; the chapter excerpt of the same name lists corrections to the library card's summary.
- [[lora-learns-less-forgets-less]] — Biderman et al., Tables S5-S13, §4.1-4.2, App. A-C: forgetting by epochs, domain, and LoRA rank.
- [[tulu-1-how-far-can-camels-go]] — Wang et al., Table 1 dataset sizes, Table 3 single-dataset versus mixture comparison at 13B, §4.2 base-model AlpacaEval note, App. D settings, App. E GSM subsample.
- [[benchmark-variance-quantified]] — Madaan et al., analytic CI formula, Table 1 seed standard deviations and CI half-widths, MMLU cloze result.
- [[adding-error-bars-evals]] — Miller, unpaired and paired standard errors (Eqs. 5-7).
- [[ifeval]] — Zhou et al., 25 instruction types, 541 prompts, strict and loose metrics.
- [[ifbench]] — Pyatkin et al., IFEval versus 58 unseen constraints.
- [[open-instruct-allenai-recipes]] — open-instruct docs: OLMo 3 template errors, loss-reduction drift, OLMo 2 1B stage times.
- [[open-instruct-allenai-recipes-recipe]] — verified OLMo-2-0425-1B-SFT recipe row.
- [[packing-position-ids]] — Kundu et al., padding-free collator mechanism and Table 2 step-count effect.
- [[hf-chat-template-assistant-mask]] — transformers v4.46.0 assistant-mask code and the OLMo 2 1B Instruct template.
- [[karpathy-training-neural-net-recipe]] — "verify loss @ init", "overfit one batch", and decoding the network's input tensors (section 2).
- [[signal-and-noise-eval]] — Heineman et al., benchmark signal-to-noise and small-subset decision accuracy in pretraining ablations.
- [[judge-llm-bias]] — Zheng et al., MT-Bench construction: 80 questions, 10 per category in 8 categories (§2.2).
