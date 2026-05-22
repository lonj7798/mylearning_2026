<!-- scope: Answer.AI QLoRA-era practical posts
     deps: [[qlora]]
     see-also: [[bitsandbytes-nf4]], [[hf-quantization-fundamentals]]
-->

# Answer.AI — FSDP + QLoRA, 70B Fine-tuning Posts
- **Core Insight:** Answer.AI's FSDP+QLoRA blog showed that you can fine-tune a 70B-parameter Llama on a pair of consumer GPUs (2× RTX 3090, 48GB) by combining QLoRA 4-bit base with FSDP sharded data parallelism — extending QLoRA's "single-GPU 65B" claim into "consumer-grade 70B".
- **Guideline:** Use FSDP-QLoRA when fine-tuning >30B models on prosumer hardware; combine NF4 4-bit base + FSDP CPU offload + LoRA adapters + paged optimizer.
- **Authors:** Jeremy Howard, Johno Whitaker, Hamel Husain, Kerem Turgutlu (Answer.AI + collaborators)
- **Year:** 2024
- **URL:** https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
- **Relevant topics:** FSDP, QLoRA, 70B fine-tuning, consumer GPUs, hybrid sharding

## Summary
Answer.AI's flagship quantization-related post is the FSDP+QLoRA writeup that demonstrated 70B-parameter fine-tuning on consumer hardware (2× RTX 3090, ~48GB combined VRAM). The breakthrough was making FSDP sharded data-parallel compatible with bitsandbytes 4-bit weights — previously the two stacks didn't interoperate cleanly because FSDP's all-gather expected FP16/BF16 parameters, while bnb stored packed INT4. Answer.AI patched `bitsandbytes` and PEFT to expose `Params4bit` correctly to FSDP and added LoRA adapter sharding. The result is a workflow where: (1) NF4 base weights are sharded across GPUs by FSDP, (2) LoRA adapters live in FP16 alongside, (3) paged optimizer offloads states to CPU memory, (4) gradient checkpointing pays for the remaining activation memory. Follow-up posts cover Llama-3 70B-on-2×4090 reproducibility and the Spectrum technique for selective layer fine-tuning.

## Key Points
- Made FSDP compatible with bitsandbytes 4-bit `Params4bit`.
- 70B Llama-2 fine-tunable on 2× RTX 3090 (24GB each).
- Combines: NF4 base + LoRA adapters (FP16) + paged optimizer + gradient checkpointing.
- Patches landed in bitsandbytes, PEFT, and Transformers main branches.
- Follow-up posts on Llama-3-70B and on Spectrum (layer-selective FT).

## Technical Details

### The original blocker
FSDP shards `nn.Parameter` tensors and all-gathers them on demand. bitsandbytes' `Params4bit` is a subclass holding packed INT4 bytes; FSDP's gather didn't know how to combine the per-shard packed bytes back into a coherent INT4 weight. Symptoms: corrupted weights after gather, NaN loss.

### The fix
1. Override `Params4bit` to expose flattened storage that FSDP can shard byte-wise.
2. Custom FSDP wrapper policy that treats `Params4bit` as a single shardable unit.
3. Re-pack the gathered bytes back into per-block 4-bit form before each forward.
4. PEFT integration patch so LoRA adapters live in FP16 alongside the sharded NF4 base.

### The full stack
```python
from accelerate import FullyShardedDataParallelPlugin
from accelerate.utils import (
    fsdp_auto_wrap_policy, BnbQuantizationConfig
)

bnb_config = BnbQuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    cpu_offload=False,           # set True if memory still tight
    auto_wrap_policy=fsdp_auto_wrap_policy,
)
# accelerate launch handles the rest
```

### Memory math (Llama-2-70B on 2× RTX 3090)
| Component | Memory |
|-----------|--------|
| NF4 base (~140GB FP16 → 35GB NF4) | 17.5 GB/GPU after FSDP shard |
| LoRA adapters (r=8) FP16 | ~0.6 GB |
| Gradient buffer | ~0.6 GB |
| Optimizer state (paged, mostly CPU) | ~1 GB GPU |
| Activations (grad checkpointing) | ~3 GB |
| **Total per GPU** | **~22 GB** (fits 24 GB) |

### Throughput
~250 tokens/sec/GPU on 2× RTX 3090 for 70B fine-tuning. Slower than A100 but accessible.

### Follow-up posts
- **Llama-3-70B on 2× RTX 4090**: same recipe, ~1.4× faster on 4090.
- **Spectrum**: layer-selective fine-tuning; only train the top-SNR layers; further memory savings.
- **GaLore integration**: alternative to LoRA using low-rank gradient projection.

### Why this matters
- Democratizes 70B fine-tuning from $50k clusters to $3k home rigs.
- Establishes the canonical "QLoRA + FSDP + paged optimizer + grad checkpointing" stack used across the open-source fine-tuning community.

## Connections
- [[qlora]] — paper this work extends.
- [[bitsandbytes-nf4]] — library patched to make it work.
- [[hf-quantization-fundamentals]] — HF posts describing the same APIs.
- [[dettmers-group]] — original QLoRA inventors who Answer.AI builds on.
