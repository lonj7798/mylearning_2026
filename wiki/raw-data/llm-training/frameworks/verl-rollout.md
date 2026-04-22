<!-- scope: rollout workers and async engine in verl
     deps: [[async-rollout]], [[ppo]]
     see-also: [[verl-ppo-loss]], [[verl-grpo]], [[openrlhf-ppo]]
-->

# verl — Rollout Loop
- **Framework:** verl (volcengine / Bytedance Seed)
- **Repo URL:** https://github.com/verl-project/verl
- **Version/commit:** `main` branch (fetched 2026-04-21)
- **Relevant file(s):**
  - `verl/workers/rollout/hf_rollout.py` ≈ lines 40–130 (`HFRollout.generate_sequences` — synchronous HF `.generate()`)
  - `verl/workers/rollout/vllm_rollout/vllm_async_server.py` ≈ lines 440–530 (`generate` async method, vLLM AsyncLLMEngine path)
  - `verl/workers/rollout/vllm_rollout/vllm_rollout.py` ≈ lines 198–214 (`ServerAdapter.generate_sequences` — now raises `NotImplementedError`; SPMD sync mode was retired in PR #4411)
- **Core pattern:** Two rollout backends. (1) `HFRollout` is the debug/reference path: FSDP `summon_full_params`, single `.generate()` call with a `GenerationConfig`, no external engine. (2) The production path is `vllm_async_server.py`: an async vLLM engine with per-request tokens-in / tokens-out, priority scheduling, LoRA adapters, MoE routing capture, and weight-transfer pause hooks for on-policy weight broadcasting.
- **Why it matters:** For LLM RL, the rollout dominates wall time (≥70% for most recipes); this file is where verl's throughput advantage over trainer-colocated HF generation actually lives.

## Context
verl decouples generation from training via a worker-group abstraction. Each rollout actor wraps a vLLM engine; the trainer holds FSDP shards; weights are broadcast from trainer→rollout between optimizer steps. The **async** server (below) is what enables overlap of rollout with the next training microbatch and is the mechanism for partial-rollout / continuous-batching RL recipes. Sync SPMD vLLM mode was removed (PR #4411) — new deployments must use async.

## Code excerpt
```python
# verl/workers/rollout/hf_rollout.py, lines 40–125 (HFRollout.generate_sequences, condensed)
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

```python
# verl/workers/rollout/vllm_rollout/vllm_async_server.py, ≈ lines 440–525 (async generate)
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

## What to notice
- **Tokens-in-tokens-out** — the async server never touches the tokenizer; prompts arrive as `prompt_ids` lists and responses come back as token ids + logprobs. Makes multi-turn tool-use loops simple.
- **Per-request priority** lets the trainer force newly-weighted requests ahead of stragglers — the knob that enables partial-rollout RL (verl's "continuous batching RL" blog post).
- **`max_tokens` clamp** has three layers: user override, global `response_length`, and context-window residual. Silent truncation is avoided by the final `assert max_tokens <= max_possible_tokens`.
- **LoRA as adapter** — when on, the trainer only broadcasts adapter weights (much smaller than full state), and the engine loads them as a vLLM LoRA request.
- **Weight broadcast pauses the engine** (`engine to paused state` block around line 628) to avoid on-the-fly weight updates during active decoding — the correctness fix for async RL.
- **FSDP path** requires `summon_full_params(..., recurse=False)` — unsharding all parameters for generation; memory-expensive but needed because HF `.generate` isn't FSDP-aware.

## Comparison to paper / to other frameworks
- **vs OpenRLHF async rollout (`openrlhf/trainer/ppo_trainer_async.py`):** OpenRLHF uses a Ray `Queue` between rollout actors and trainer + a `Lock` around vLLM weight broadcast; verl uses async-vLLM's internal scheduler + priority + explicit pause state. Same architectural pattern.
- **vs TRL `_generate_vllm_server` / `_generate_vllm_colocate`:** TRL's in-process vLLM integration is closer to verl's old SPMD path; no partial-rollout or priority knobs.
- **vs HF `.generate` directly:** `HFRollout` is a thin wrapper for correctness-testing only; production verl runs async vLLM.
