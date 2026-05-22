<!-- scope: DeepSpeed-FastGen / Dynamic SplitFuse inference system source page
     deps: [[continuous-batching]]
     see-also: [[vllm]], [[tensorrt-llm]]
-->

# DeepSpeed-FastGen
- **Core Insight:** DeepSpeed-FastGen improves LLM serving throughput and latency with Dynamic SplitFuse, which decomposes prompt and generation work into schedulable chunks.
- **Guideline:** Treat FastGen as a source for scheduling ideas and DeepSpeed inference integration; verify current maintenance before choosing it over newer serving engines.
- **Authors:** Microsoft DeepSpeed team
- **Year:** 2023-2024
- **URL:** https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md
- **Relevant topics:** Dynamic SplitFuse, blocked KV cache, prompt/decode scheduling, DeepSpeed-Inference, tensor parallelism

## Abstract
DeepSpeed-FastGen is a DeepSpeed inference system and blog/release track focused on high-throughput text generation. Its central technique, Dynamic SplitFuse, splits long prompts into smaller chunks and fuses them with decode work from other requests. The design targets better latency/throughput balance than batching entire prompt phases separately from generation.

## Key Contributions
- Introduces Dynamic SplitFuse scheduling for mixed prompt and generation workloads.
- Uses a blocked KV cache design to support flexible request scheduling.
- Integrates with DeepSpeed-Inference and tensor parallel serving.
- Provides examples for Hugging Face text generation using DeepSpeed.
- Frames scheduling around avoiding both prompt monopolization and decode underutilization.

## Key Figures/Tables to Study
- DeepSpeed-FastGen blog README: Dynamic SplitFuse diagrams and benchmark claims.
- DeepSpeed inference examples: launch/configuration pattern.
- DeepSpeed inference source: kernel injection, tensor parallel, and generation runtime context.
- Any FastGen examples in DeepSpeedExamples for Hugging Face text generation.

## Technical Details
Public APIs:
- Users interact through DeepSpeed inference configuration and example scripts rather than a standalone OpenAI-compatible server.
- Example serving/generation paths are in DeepSpeedExamples and DeepSpeed-Inference docs.

Scheduler/cache approach:
- Dynamic SplitFuse breaks prompt prefill into chunks.
- Chunks are scheduled together with generation/decode tokens from other requests.
- This reduces stalls where decode requests wait behind a large prefill.
- Blocked KV cache supports non-monolithic cache management for active requests.

Relevant code/docs:
- FastGen blog source: https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepspeed-fastgen/README.md
- DeepSpeed repo: https://github.com/deepspeedai/DeepSpeed
- DeepSpeedExamples text generation: https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/text-generation
- DeepSpeed inference docs: https://www.deepspeed.ai/inference/

Strengths:
- Important scheduling reference for chunked prefill and prompt/decode fusion.
- Fits users already invested in DeepSpeed training/inference workflows.
- Useful historical bridge between static batching and modern continuous batching systems.

Limitations:
- Not as dominant as vLLM/SGLang/TensorRT-LLM for current standalone LLM serving.
- Public docs emphasize the blog/design more than a maintained server API.
- Integration details may require reading DeepSpeed examples and version-specific code.

## Connections
- Dynamic SplitFuse relates to chunked prefill in [[vllm-scheduler]] and [[sglang-scheduler]].
- Blocked KV cache connects to [[pagedattention]] and [[tensorrt-llm-paged-kv]].
- Compare as a scheduling design source rather than a general OpenAI-compatible server like [[hf-tgi]].
