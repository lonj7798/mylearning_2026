<!-- scope: LMDeploy and TurboMind serving architecture source page
     deps: [[continuous-batching]]
     see-also: [[tensorrt-llm]], [[vllm]]
-->

# LMDeploy
- **Core Insight:** LMDeploy serves LLMs through TurboMind and PyTorch backends, with persistent batching and an extendable KV-cache manager as key TurboMind ideas.
- **Guideline:** Use LMDeploy when its supported models/backends fit and you want a practical OpenAI-compatible server with TurboMind optimizations.
- **Authors:** InternLM / OpenMMLab contributors
- **Year:** 2023-present
- **URL:** https://lmdeploy.readthedocs.io/ and https://github.com/InternLM/lmdeploy
- **Relevant topics:** TurboMind, persistent batch, KV cache manager, OpenAI server, paged attention, FasterTransformer lineage

## Abstract
LMDeploy is a toolkit for compression, deployment, and serving of LLMs. Its TurboMind backend is an inference engine for conversational LLMs with a persistent-batch execution model and an extendable KV-cache manager. LMDeploy also provides a PyTorch engine, OpenAI-compatible server, RESTful APIs, chat tools, quantization, and multimodal serving.

## Key Contributions
- TurboMind models serving as a persistent batch whose lifetime spans the serving process.
- KV cache manager acts like a memory pool and LRU cache for reusable conversation KV states.
- OpenAI-compatible server under `lmdeploy serve openai` and related modules.
- PyTorch backend includes paging scheduler and paged-attention kernels.
- Supports quantization and deployment workflows around InternLM and other common model families.

## Key Figures/Tables to Study
- TurboMind architecture docs: API, persistent batch, KV cache manager, and LLaMA implementation diagram.
- `lmdeploy/serve/openai`: public server and protocol handling.
- `lmdeploy/pytorch/paging/scheduler.py`: PyTorch backend paging scheduler.
- `lmdeploy/pytorch/engine/cache_engine.py`: cache engine for PyTorch path.
- `lmdeploy/pytorch/kernels/cuda/pagedattention.py`: paged attention kernel wrapper.

## Technical Details
Public APIs:
- CLI server commands under `lmdeploy serve`, including OpenAI-compatible API server.
- Python pipeline/chat APIs for local inference.
- Backend configuration chooses TurboMind or PyTorch paths depending on model/use case.

Scheduler/cache approach:
- TurboMind persistent batch preallocates batch slots and lets requests join when slots are free.
- KV cache manager can preserve/reuse conversation history and evict using LRU-like behavior.
- PyTorch backend has explicit paging scheduler and cache engine modules.
- Cache hits can skip repeated history decoding and start response generation sooner.

Relevant code/docs:
- Docs: https://lmdeploy.readthedocs.io/
- TurboMind architecture: https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html
- GitHub: https://github.com/InternLM/lmdeploy
- OpenAI server source: https://github.com/InternLM/lmdeploy/tree/main/lmdeploy/serve/openai
- PyTorch paging source: https://github.com/InternLM/lmdeploy/tree/main/lmdeploy/pytorch/paging

Strengths:
- Clear architecture docs for persistent batch and KV-cache manager.
- Practical OpenAI-compatible serving plus optimized backend options.
- Useful bridge between FasterTransformer-style engines and modern paged serving designs.

Limitations:
- Documentation can differ by version; older TurboMind docs remain important for architecture.
- Backend choice affects feature availability and tuning.
- Ecosystem is narrower than Hugging Face/vLLM for arbitrary model serving.

## Connections
- Persistent batch is LMDeploy's framing of [[continuous-batching]].
- KV manager comparisons: [[vllm-kv-cache-manager]], [[sglang-radixattention]], and [[tensorrt-llm-paged-kv]].
- OpenAI serving surface compares with [[hf-tgi]] and [[llama-cpp-server]].
