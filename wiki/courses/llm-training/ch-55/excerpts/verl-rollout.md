---
chapter: ch-55
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/frameworks/verl-rollout.md
source_url: https://github.com/verl-project/verl
created_at: "2026-04-23"
---

# Excerpt: verl Rollout — HFRollout + async vLLM server

**Source library:** `wiki/raw-data/llm-training/frameworks/verl-rollout.md`
**Artifact:** `HFRollout.generate_sequences` (reference) + `vllm_async_server.py::generate` (production). The retired `ServerAdapter.generate_sequences` (PR #4411).

---

## Why this excerpt exists in ch-55

Ch-55 §4 compares the two rollout backends. Rollout dominates RL wall-clock (≥70% typical). The file path that actually runs at 70B scale is `vllm_async_server.py`; the file you read first for correctness debugging is `hf_rollout.py`. The third backend — sync SPMD vLLM — was retired by PR #4411 and now raises.

---

## HFRollout — the reference path

```python
# verl/workers/rollout/hf_rollout.py, lines 40-125 (HFRollout.generate_sequences, condensed)
class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        return DataProto.concat(output)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        kwargs = ({"do_sample": False, "num_beams": 1} if not do_sample
                  else {"do_sample": True, "num_beams": 1, "top_p": top_p, "top_k": top_k,
                        "temperature": temperature, "num_return_sequences": 1})
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx, attention_mask=attention_mask,
                do_sample=do_sample, max_new_tokens=response_length,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False, return_dict_in_generate=True, use_cache=True,
            )
        seq = output.sequences
        # ... pad/truncate to response_length, build response mask, return DataProto
```

**Why this is correctness-only:** `FSDP.summon_full_params(..., recurse=False)` unshards every parameter to every rank — fine on 1B × 1 GPU, unusable at 70B × 64 GPUs. Logprob parity with training is exact because the same forward pass is used. Use it to localize numerical divergences between training and production rollout.

---

## Async vLLM server — the production path

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py, ~lines 440-525
async def generate(self, prompt_ids, sampling_params, request_id,
                   image_data=None, video_data=None, priority=0) -> TokenOutput:
    prompt_ids = normalize_token_ids(prompt_ids)
    max_possible_tokens = self.config.max_model_len - len(prompt_ids)
    if   "max_tokens"     in sampling_params: max_tokens = sampling_params.pop("max_tokens")
    elif "max_new_tokens" in sampling_params: max_tokens = sampling_params.pop("max_new_tokens")
    else: max_tokens = min(
        self.config.response_length,
        self.config.prompt_length + self.config.response_length - len(prompt_ids),
    )
    max_tokens = max(0, min(max_tokens, max_possible_tokens))
    sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
    sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
    sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

    prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)
    lora_request = LoRARequest(VLLM_LORA_NAME, VLLM_LORA_INT_ID, VLLM_LORA_PATH) \
                   if self.lora_as_adapter and VLLM_LORA_INT_ID in await self.engine.list_loras() \
                   else None
    generator = self.engine.generate(
        prompt=prompt, sampling_params=sampling_params,
        request_id=request_id, lora_request=lora_request, priority=priority,
    )
    final_res = None
    async for output in generator:
        final_res = output
    token_ids = final_res.outputs[0].token_ids
    log_probs = ([logprobs[token_ids[i]].logprob
                  for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
                 if sampling_params.logprobs is not None else None)
    return TokenOutput(token_ids=token_ids, log_probs=log_probs, ...)
```

**Six production levers** from the source's "What to notice":

1. **Tokens-in-tokens-out.** No tokenizer in the server; prompts arrive as `prompt_ids`, responses come back as token ids + logprobs. Makes multi-turn tool-use loops simple.
2. **`priority`.** Per-request ordering; lets the trainer force newly-re-weighted requests ahead of stragglers. The knob that enables partial-rollout RL (verl's "continuous batching RL" blog).
3. **3-layer `max_tokens` clamp.** User override → global `response_length` → context-window residual. `assert max_tokens <= max_possible_tokens` avoids silent truncation.
4. **LoRA as adapter.** Broadcast only adapter weights (MB, not GB); engine loads them as a vLLM `LoRARequest`.
5. **Paused-state weight broadcast** (~line 628). Blocks new generates and waits for in-flight to drain before a weight update — correctness fix for async RL.
6. **FSDP path requires `summon_full_params`** only on `HFRollout`; the async server keeps its own sharded weights, broadcast from the trainer.

---

## The retired SPMD path — PR #4411

```python
# verl/workers/rollout/vllm_rollout/vllm_rollout.py, ~lines 198-214
class ServerAdapter:
    def generate_sequences(self, prompts):
        raise NotImplementedError(
            "SPMD sync vLLM mode was retired in PR #4411; use AsyncLLMEngine via "
            "vllm_async_server.py"
        )
```

Why retired: SPMD colocated the engine with the trainer → no per-request priority (no partial rollout), no paused-state weight broadcast (decoding could run on mixed weights), and throughput hit a ceiling at ~2× HFRollout. Async lifts all three limits.

---

## Comparison table (ch-55 §4.3 verbatim)

| Concern                | HFRollout                  | async vLLM server                    |
|------------------------|----------------------------|--------------------------------------|
| Throughput (7B, 8×H100)| ~1× (baseline)             | ~5–8× (continuous batching)          |
| FSDP aware             | No (`summon_full_params`)  | Yes (separate worker, own weights)   |
| Partial rollout        | No                         | Yes (via `priority`)                 |
| Weight broadcast       | In-place                   | Paused-state + IS correction needed  |
| Train-rollout logprob  | Exact (same forward)       | Drift (`vllm_kl` metric needed)      |
| Multi-turn tool use    | Awkward (retokenize)       | Natural (tokens-in-tokens-out)       |
| Use it when            | Debugging numerical bug    | Everything else                      |

---

## Connections

- [[async-rollout]] — HybridFlow architecture, IMPALA lineage, IS correction on bounded-k staleness.
- [[verl-ppo-loss]] — `rollout_is_weights` argument is what reconciles the async path with the PPO ratio.
- OpenRLHF equivalent: `openrlhf/trainer/ppo_trainer_async.py` — Ray `Queue` + `vllm_lock` instead of priority + paused-state.
