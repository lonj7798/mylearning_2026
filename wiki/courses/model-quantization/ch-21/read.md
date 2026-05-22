<!-- chapter: ch-21
     track: eval-lab-capstone
     kind: lab
     title: Lab — Quantize an Open Model End-to-End
     deps: [ch-08, ch-09, ch-12, ch-19, ch-20]
     sources: [[awq]], [[gptq]], [[qlora]], [[marlin-kernel]], [[vllm-quant]], [[autogptq]], [[autoawq]], [[bitsandbytes-nf4]]
-->

# Chapter 21 — Lab: Quantize an Open Model End-to-End

> **Lab objective.** Produce a head-to-head Pareto comparison of four W4-class PTQ methods (AutoGPTQ-W4, AutoAWQ-W4, QLoRA-NF4, bitsandbytes-INT8) on a single open LLM, evaluated under the full [[ch-20]] harness (PPL + 5 task evals + sensitivity ablation), served through vLLM with the production kernel for each format, and shipped as a one-page memo your peer can reproduce from your repo. Pick **one** required ablation (group_size for GPTQ, alpha for AWQ, or LoRA rank for QLoRA) and report its effect.
>
> **Guideline.** Two paths are offered — a *full-budget* path on Llama-3-8B + H100, and a *resource-constrained* path on Qwen2.5-1.5B (or 1.8B) + single 16 GB GPU. The framework and discipline are identical; only the scale differs. Whichever path you take, the memo must include the side-by-side Pareto table, the calibration spec (per [[ch-20]] §5.4), and one *specific* failure mode you found — not a generic "AWQ was faster than GPTQ."

---

## Goal — three artifacts

1. **A repo.** `quant-lab/` with one quantize script per method, one `evaluate.py` covering the full harness, and a `Makefile` that reproduces the entire lab from clean.
2. **A results table.** `results.json` with one row per `(method, model, hyperparameter)` cell + 5-seed CIs on PPL and MMLU.
3. **A memo.** `lab-memo.md`, one page: Pareto table + ablation result + one specific failure mode + one Pareto-frontier recommendation per workload class (latency-bound / memory-bound / fine-tune-required).

Reproducibility per [[ch-20]] §5.4: the calibration set, tokenizer version, seeds, and hardware are pinned in the repo, not in the memo prose.

---

## Full-budget path

**Target.** 1 × H100 (80 GB) or 8 × A100-40 GB, Llama-3-8B base, ~$100 of compute, ~6 hours wall-clock end-to-end.

**Model.** `meta-llama/Llama-3-8B` as the FP16 baseline. If unavailable in your environment, `Meta-Llama-3-8B-Instruct` or `Llama-3.1-8B` are valid substitutes — record the substitution in the memo.

**Methods.** Four quantization methods at W4 (or W8 for the bitsandbytes-INT8 baseline):

| Method | Spec | Framework | Reference |
|--------|------|-----------|-----------|
| **AutoGPTQ-W4** | INT4 + group_size 128 + act_order | [[autogptq]] | [[gptq]] |
| **AutoAWQ-W4** | INT4 + group_size 128 + α grid | [[autoawq]] | [[awq]] |
| **QLoRA-NF4** | NF4 + double-quant + LoRA r=64 (fine-tune step) | [[bitsandbytes-nf4]] | [[qlora]] |
| **bitsandbytes-INT8** | INT8 + LLM.int8-style outlier-FP16 | bnb `load_in_8bit=True` | LLM.int8 |
| **FP16 baseline** | the unmodified base | HF transformers | — |

**Calibration set.** Pinned, identical across methods:
- Corpus: `allenai/c4` English `validation` split.
- Size: 128 sequences × 2048 tokens.
- Seed: 42 (`random.sample` then HF tokenizer).
- Mix: 100% C4 for the headline run; an alternate 50/50 (C4 + UltraChat) is optional for the chat-distribution sanity check.

**Eval harness.** Run per [[ch-20]] §6:
- **PPL**: Wikitext-2 + C4 (5-seed CI on both).
- **MMLU** (5-shot) — knowledge.
- **GSM8K** (8-shot CoT) — math.
- **HumanEval** (0-shot) — code.
- **IFEval** (strict) — instruction following.
- **TruthfulQA** (multiple-choice) — calibration.
- **Sensitivity ablation**: per-component sensitivity map (skip if running short on time, but state the omission in the memo).

**Inference.** Serve through vLLM (see [[vllm-quant]]):
- AutoGPTQ checkpoint → `LLM(model_path, quantization="gptq_marlin")` → Marlin kernel ([[marlin-kernel]]).
- AutoAWQ checkpoint → `LLM(model_path, quantization="awq_marlin")`.
- bitsandbytes / QLoRA → `LLM(model_path, quantization="bitsandbytes")` (vLLM bnb support; QLoRA-trained LoRA weights merged or attached via adapter API).
- FP16 baseline → no `quantization=` flag.

Measure decode latency at batch 1 / 16 / 64; peak VRAM during serving; tokens/sec at batch 16.

**Wall-clock budget.**
- AutoGPTQ quantize (8B, 128 seq × 2048 tok): ~30 min on 1×H100.
- AutoAWQ quantize: ~15 min.
- QLoRA: depends on fine-tune target — for the lab, do a 1-epoch fine-tune on 5K Alpaca samples (~30 min).
- bitsandbytes-INT8 load: <2 min (no calibration step).
- Eval harness: ~2 hours per method (5 evals × 5 seeds).
- vLLM benchmark: ~15 min per method.

---

## Resource-constrained path

**Target.** 1 × consumer GPU (≥ 16 GB; RTX 3090 / 4090 / A4000 all viable), `Qwen2.5-1.5B-Instruct` (or `Qwen2.5-Math-1.5B`), ~4 hours wall-clock.

**Model.** `Qwen/Qwen2.5-1.5B-Instruct` as the FP16 baseline. Alternative: `Qwen/Qwen2.5-3B-Instruct` if you have 24 GB.

**Methods.** Same four methods; deltas shrink at small scale but cascade shapes are identical.

**Calibration set.** 64 sequences × 1024 tokens (half the full-budget spec — small model needs less). Otherwise identical pinning.

**Eval harness.** Subset:
- **PPL**: Wikitext-2 only (5-seed CI).
- **MMLU** (5-shot) + **GSM8K** (8-shot CoT) — the two highest-signal academic tasks.
- **HumanEval** (0-shot) — code.
- Skip TruthfulQA, IFEval (less signal at 1.5B).
- Sensitivity ablation: optional.

**Inference.** Serve through vLLM with the same kernel selection. Measure decode latency at batch 1 + 16 only; batch 64 likely OOM at 16 GB.

**Wall-clock budget.**
- Per-method quantize: 5–15 min.
- Per-method eval: ~30 min.
- vLLM benchmark: ~10 min per method.

The trade-off: smaller PPL/MMLU deltas (1.5B has less headroom for quantization to hurt), but the *methodology* is the deliverable, not the absolute numbers.

---

## Step-by-step: AutoGPTQ-W4

Concrete commands. Adjust paths to your repo layout.

```bash
# install
pip install auto-gptq optimum

# quantize
python -c "
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import random

MODEL = 'meta-llama/Llama-3-8B'  # or Qwen/Qwen2.5-1.5B-Instruct
tok = AutoTokenizer.from_pretrained(MODEL)

# Calibration set — per ch-20 §5.4
random.seed(42)
c4 = load_dataset('allenai/c4', 'en', split='validation', streaming=False)
samples = random.sample(list(c4), 256)[:128]  # 128 sequences
calib = [tok(s['text'], return_tensors='pt', max_length=2048, truncation=True)
         for s in samples]
calib = [{'input_ids': c.input_ids, 'attention_mask': c.attention_mask} for c in calib]

qcfg = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,       # = actorder=True (ch-08)
    damp_percent=0.01,   # = percdamp (ch-08)
    sym=False,           # asymmetric — see ch-20 §2.3
)
model = AutoGPTQForCausalLM.from_pretrained(MODEL, qcfg)
model.quantize(calib)
model.save_quantized('out/llama3-8b-gptq-w4-g128')
tok.save_pretrained('out/llama3-8b-gptq-w4-g128')
"

# serve with Marlin
python -c "
from vllm import LLM, SamplingParams
llm = LLM('out/llama3-8b-gptq-w4-g128', quantization='gptq_marlin', dtype='auto')
print(llm.generate(['Hello'], SamplingParams(max_tokens=20))[0].outputs[0].text)
"
```

See [[autogptq]] for the exact CLI of newer GPTQModel-fork.

---

## Step-by-step: AutoAWQ-W4

```bash
pip install autoawq

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL = 'meta-llama/Llama-3-8B'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoAWQForCausalLM.from_pretrained(MODEL, device_map='auto')

# Calibration set — same pinning as GPTQ
import random
from datasets import load_dataset
random.seed(42)
c4 = load_dataset('allenai/c4', 'en', split='validation')
samples = random.sample(list(c4), 256)[:128]
calib_data = [s['text'] for s in samples]

qcfg = {
    'w_bit': 4,
    'q_group_size': 128,
    'zero_point': True,   # asymmetric — see ch-20 §2.3
    'version': 'GEMM',
}
model.quantize(tok, quant_config=qcfg, calib_data=calib_data, max_calib_seq_len=512)
model.save_quantized('out/llama3-8b-awq-w4-g128')
tok.save_pretrained('out/llama3-8b-awq-w4-g128')
"

# serve with AWQ-Marlin
python -c "
from vllm import LLM
llm = LLM('out/llama3-8b-awq-w4-g128', quantization='awq_marlin', dtype='auto')
"
```

Note: AutoAWQ's `max_calib_seq_len=512` is the [[awq]] default — shorter than GPTQ's 2048 because AWQ saturates faster (one scalar per layer, not a full Hessian).

---

## Step-by-step: QLoRA-NF4 (fine-tune workflow)

QLoRA is not purely a quantization step — it ships an NF4-quantized base and trains a LoRA adapter on top. The W4-equivalent comparison is: take the NF4-quantized base, fine-tune a LoRA on Alpaca-style data, then merge or attach the adapter for serving.

```bash
pip install bitsandbytes peft trl

python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL = 'meta-llama/Llama-3-8B'
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# NF4 + double-quant per QLoRA paper (ch-12)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map='auto')

# LoRA per QLoRA defaults
lcfg = LoraConfig(
    r=64,                  # ← required ablation knob (try 4 / 16 / 64)
    lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0.0,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lcfg)

# Fine-tune on Alpaca-style for ~5K samples / 1 epoch (~30 min on H100)
ds = load_dataset('tatsu-lab/alpaca', split='train').select(range(5000))

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(
        output_dir='out/llama3-8b-qlora-r64',
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type='constant',
        optim='paged_adamw_32bit',
        bf16=True,
        save_strategy='epoch',
    ),
)
trainer.train()
trainer.save_model('out/llama3-8b-qlora-r64')
"
```

For evaluation, either (a) keep the LoRA attached at serve time (`vLLM` supports LoRA adapters since 0.5+), or (b) merge the LoRA into the NF4 base and reload as a standalone model. Option (a) is the realistic deployment shape.

See [[bitsandbytes-nf4]] for the underlying kernel details and [[qlora]] for the recipe context.

---

## Step-by-step: bitsandbytes-INT8

The simplest baseline — no calibration, just a load flag.

```bash
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'meta-llama/Llama-3-8B'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, load_in_8bit=True, device_map='auto',
)
# That's it. Save not strictly needed — bnb is a load-time transform.
"
```

For vLLM serving, use `quantization='bitsandbytes'` — vLLM re-runs the bnb 8-bit transform on load.

---

## The required ablation

Pick **one** of the following and report the effect:

### Option A — GPTQ `group_size`

Run AutoGPTQ-W4 with `group_size ∈ {32, 64, 128, 256}`. Report PPL + MMLU + effective-bits/parameter for each. Expected shape: PPL improves monotonically with smaller group_size; effective bits/weight grows. The Pareto question: at which `group_size` does the marginal accuracy gain fall below the marginal storage cost? (Standard answer: 128 is the production sweet spot; 32 wins on quality at ~5% more storage.)

### Option B — AWQ `α` grid extension

AutoAWQ's default grid is 20 points in [0, 1]. Disable the grid search and force `α ∈ {0.3, 0.5, 0.7}` to characterise the bowl shape. Plot PPL vs α. The bowl minimum is a function of the model — usually around 0.5 — but watching the bowl helps you internalise why AWQ's grid search is robust.

### Option C — QLoRA LoRA rank

Run QLoRA-NF4 with `r ∈ {4, 8, 16, 32, 64}`. Report PPL + MMLU + adapter parameter count + fine-tune wall-clock. Expected shape: quality improves through r=16 or 32, plateaus by 64; wall-clock grows ~linearly. The Pareto question: is the QLoRA quality gain worth the fine-tune compute relative to the calibration-only methods? (Hint: QLoRA *wins* on instruction-following / chat quality if the SFT data is good; *loses* on raw pretraining PPL.)

The required ablation is what turns this lab from a "I ran four methods" exercise into a "I learned something about a method" exercise.

---

## The Pareto deliverable

The headline table. One row per `(method, model)` cell:

| Method | Bits/wt eff | PPL Wikitext-2 ± CI | MMLU ± CI | GSM8K | HumanEval | IFEval | TruthfulQA | Latency bs=1 (ms/tok) | Latency bs=16 (ms/tok) | Peak VRAM (GB) | Quantize wall-clock |
|--------|------------|--------------------|-----------|-------|-----------|--------|------------|----------------------|-----------------------|---------------|----------------------|
| FP16 baseline | 16.0 | x ± y | x ± y | x | x | x | x | x | x | x | — |
| AutoGPTQ-W4 g128 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~30 min |
| AutoAWQ-W4 g128 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~15 min |
| QLoRA-NF4 r=64 | 4.13 | x ± y | x ± y | x | x | x | x | x | x | x | ~30 min |
| bitsandbytes-INT8 | 8.5 | x ± y | x ± y | x | x | x | x | x | x | x | <2 min |

Plus a Pareto chart: quality (e.g., MMLU) on Y, latency on X, with a marker per method. The Pareto frontier is the lower-right boundary; methods strictly dominated by another method on both axes are *off* the frontier.

---

## Reflection prompts

The memo concludes with answers to these. The answers separate "did the lab" from "learned from the lab."

1. **Which method is on the Pareto frontier for which workload class?** Latency-bound batch-1 chat? Memory-bound long-context? Fine-tune-required domain adaptation? You should have one method per class.
2. **When does fine-tunability dominate raw speedup?** QLoRA is slower than AWQ at inference and worse at PPL. When does its tunability still make it the right pick?
3. **Where does AWQ generalise better than GPTQ in your numbers?** [[awq]] claims robustness across domains and modalities because it doesn't backprop into reconstruction. Did your numbers reproduce this on the OOD task (e.g., HumanEval if calibration was C4)?
4. **What did the required ablation teach you?** State the result and what surprised you. If nothing surprised you, you ran the wrong ablation or missed the surprise.
5. **One specific failure mode.** Not "AWQ was slower than I expected." A specific failure: "GPTQ-W4 group_size=128 lost 4 points on IFEval-strict because the per-128-channel scale collapsed the LayerNorm gain dependency on the format-token paths." See [[ch-20]] §1 for the kinds of pathologies to look for.

---

## What this lab is not

- **Not a benchmark of quantization libraries.** AutoGPTQ vs AutoAWQ vs bitsandbytes is a comparison of *methods*, not *libraries*. If you find a 1.3× speed difference between AutoGPTQ's CUDA kernel and Marlin, that's a kernel-engineering finding, not a method finding.
- **Not a sensitivity-analysis lab.** That's the optional component-sensitivity ablation. The lab's *required* deliverable is the four-method Pareto + one knob ablation.
- **Not the capstone.** [[ch-22]] asks you to reproduce a frontier paper from the paper alone — a different exercise. This lab is the warm-up: a fixed harness, four known-good methods, and one knob.

The point of the lab is *fluency*. After it, you should be able to quantize a new open model with any of these four methods in <2 hours of wall-clock, and you should be able to read a quantization paper and know exactly which of the four it competes with.

---

## Connections

- **Back to [[gptq]] (ch-08)** — the algorithm AutoGPTQ implements.
- **Back to [[awq]] (ch-09)** — the algorithm AutoAWQ implements.
- **Back to [[qlora]] (ch-12)** — the NF4 + LoRA recipe.
- **Back to [[ch-19]]** — production kernels (Marlin / Machete / vLLM serving).
- **Back to [[ch-20]]** — the evaluation methodology this lab applies.
- **Forward to [[ch-22]]** — the capstone where you reproduce a *paper*, not a known recipe.
- [[autogptq]] / [[autoawq]] / [[bitsandbytes-nf4]] — implementation references.
- [[vllm-quant]] / [[marlin-kernel]] — serving stack.

## Excerpts

- [[excerpts/lab-recipe]] — the consolidated four-method execution recipe with all commands in one place.
- [[excerpts/ablation-design]] — design notes for the three ablation options and how to interpret the result.
- [[excerpts/pareto-memo-template]] — the headline table template + reflection scaffolding.
- [[excerpts/serving-stack]] — vLLM kernel selection per checkpoint format and the latency-measurement protocol.
