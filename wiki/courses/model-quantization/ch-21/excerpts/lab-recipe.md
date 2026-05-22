---
chapter: ch-21
course: model-quantization
phase: read
excerpt_of: "Consolidated four-method execution recipe for the ch-21 lab"
created_at: "2026-05-21"
---

# Excerpt: Lab Execution Recipe

**Sources:** [[raw-data/awq]], [[raw-data/gptq]], [[raw-data/qlora]], [[raw-data/autogptq]], [[raw-data/autoawq]], [[raw-data/bitsandbytes-nf4]], [[raw-data/vllm-quant]]

---

## Repo layout (recommended)

```
quant-lab/
├── Makefile                 # one-command reproduction
├── requirements.txt         # pinned versions of vllm, autoawq, auto-gptq, bnb, peft, trl
├── config.py                # MODEL / CALIB_SOURCE / SEED / OUT_DIR constants
├── quantize/
│   ├── gptq.py              # AutoGPTQ-W4
│   ├── awq.py               # AutoAWQ-W4
│   ├── qlora.py             # NF4 + LoRA fine-tune
│   └── int8.py              # bitsandbytes-INT8
├── evaluate.py              # full ch-20 harness across all checkpoints
├── benchmark.py             # vLLM latency + VRAM + throughput
├── ablation.py              # required-ablation runner (one of group_size / alpha / lora_rank)
└── memo.md                  # output: the one-page Pareto memo
```

The `Makefile` should run end-to-end with:
```
make all   # quantize + eval + benchmark + memo, in that order
```

Anything that requires manual intervention belongs as a comment in the Makefile, not as a step in the memo.

---

## Pinned constants (`config.py`)

```python
MODEL = "meta-llama/Llama-3-8B"             # full-budget
# MODEL = "Qwen/Qwen2.5-1.5B-Instruct"       # resource-constrained

CALIB_SOURCE = "allenai/c4"
CALIB_SPLIT = "validation"
CALIB_N_SEQ = 128                            # 64 for small model
CALIB_TOK_LEN = 2048                         # 1024 for small model
SEED = 42

EVAL_SEEDS = [42, 43, 44, 45, 46]            # 5 seeds for CI

OUT_DIR = "out"
```

Every quantize script imports from `config.py`. No magic numbers in the scripts. This is the [[ch-20]] §5.4 reproducibility discipline applied.

---

## Shared calibration loader

```python
# quantize/_calib.py
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL, CALIB_SOURCE, CALIB_SPLIT, CALIB_N_SEQ, CALIB_TOK_LEN, SEED

def load_calibration():
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset(CALIB_SOURCE, "en", split=CALIB_SPLIT)
    # Reproducible sample: random.sample on a deterministically-ordered subset
    pool = list(ds.shuffle(seed=SEED).select(range(2048)))
    samples = random.sample(pool, CALIB_N_SEQ)
    texts = [s["text"] for s in samples]
    return tok, texts, samples
```

This is the single source of truth for "the calibration set." If two scripts use different calibration data, the comparison across methods is not apples-to-apples.

---

## The four quantize scripts (skeletons)

### `quantize/gptq.py`

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from _calib import load_calibration
from config import MODEL, OUT_DIR

tok, texts, _ = load_calibration()
calib = [tok(t, return_tensors='pt', max_length=2048, truncation=True)
         for t in texts]
calib = [{'input_ids': c.input_ids, 'attention_mask': c.attention_mask}
         for c in calib]

qcfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True,
                         damp_percent=0.01, sym=False)
m = AutoGPTQForCausalLM.from_pretrained(MODEL, qcfg)
m.quantize(calib)
m.save_quantized(f"{OUT_DIR}/gptq-w4-g128")
tok.save_pretrained(f"{OUT_DIR}/gptq-w4-g128")
```

### `quantize/awq.py`

```python
from awq import AutoAWQForCausalLM
from _calib import load_calibration
from config import MODEL, OUT_DIR

tok, texts, _ = load_calibration()
m = AutoAWQForCausalLM.from_pretrained(MODEL, device_map='auto')
m.quantize(tok,
           quant_config={'w_bit':4, 'q_group_size':128,
                         'zero_point':True, 'version':'GEMM'},
           calib_data=texts, max_calib_seq_len=512)
m.save_quantized(f"{OUT_DIR}/awq-w4-g128")
tok.save_pretrained(f"{OUT_DIR}/awq-w4-g128")
```

### `quantize/qlora.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from config import MODEL, OUT_DIR

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                       bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.bfloat16)
m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                        device_map='auto')

lcfg = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.0, bias='none',
                  task_type='CAUSAL_LM',
                  target_modules=['q_proj','k_proj','v_proj','o_proj',
                                  'gate_proj','up_proj','down_proj'])
m = get_peft_model(m, lcfg)

ds = load_dataset('tatsu-lab/alpaca', split='train').select(range(5000))
SFTTrainer(model=m, tokenizer=tok, train_dataset=ds,
           args=SFTConfig(output_dir=f"{OUT_DIR}/qlora-r64",
               num_train_epochs=1, per_device_train_batch_size=4,
               gradient_accumulation_steps=4, learning_rate=2e-4,
               lr_scheduler_type='constant', optim='paged_adamw_32bit',
               bf16=True)).train()
```

### `quantize/int8.py`

```python
# bitsandbytes-INT8 is a load-time transform — there is no save step.
# This file just verifies the load works; serving re-applies bnb at runtime.
import torch
from transformers import AutoModelForCausalLM
from config import MODEL
_ = AutoModelForCausalLM.from_pretrained(MODEL, load_in_8bit=True,
                                       device_map='auto')
print("INT8 load OK")
```

---

## Why these defaults

Every default in the four scripts is traceable to a paper or framework convention:

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| GPTQ `group_size=128` | [[gptq]] standard recipe |
| GPTQ `desc_act=True` | [[gptq]] act_order trick, +0.1-0.3 PPL |
| GPTQ `damp_percent=0.01` | [[gptq]] standard |
| GPTQ `sym=False` | [[ch-20]] §2.3 — asymmetric for instruction-tuned base |
| AWQ `q_group_size=128` | [[awq]] standard |
| AWQ `zero_point=True` | [[ch-20]] §2.3 — asymmetric |
| AWQ `version='GEMM'` | [[autoawq]] — best for prefill/eval workloads |
| AWQ `max_calib_seq_len=512` | [[awq]] — AWQ saturates faster than GPTQ |
| QLoRA `bnb_4bit_quant_type='nf4'` | [[qlora]] — quantile-fit for Gaussian weights |
| QLoRA `bnb_4bit_use_double_quant=True` | [[qlora]] — save 0.37 bits/weight |
| QLoRA `r=64`, `lora_alpha=16` | [[qlora]] Guanaco recipe |
| QLoRA `target_modules = all linears` | [[qlora]] — not just attention; gate/up/down too |
| QLoRA `optim='paged_adamw_32bit'` | [[qlora]] paged optimizer |
| QLoRA `learning_rate=2e-4`, `constant` | [[qlora]] Guanaco recipe |

If you change any of these, document the change and the rationale in the memo.

---

## Connections

- [[ch-21]] §step-by-step — chapter sections this excerpt expands.
- [[autogptq]] / [[autoawq]] / [[bitsandbytes-nf4]] — implementation references.
- [[gptq]] / [[awq]] / [[qlora]] — algorithm papers.
- [[vllm-quant]] — serving stack used for benchmarking.
