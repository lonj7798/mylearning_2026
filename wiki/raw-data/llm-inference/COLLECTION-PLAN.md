<!-- scope: master checklist of topics + target sources for llm-inference raw library
     deps: [[README]]
     see-also: [[insights]]
-->

# Collection Plan — Topic Checklist (LLM Inference)

Every bullet below is a target file. Mark `[x]` when the file lands in the right subdirectory with the required structure (see `[[README]]`). Gaps are filled in a second sweep after initial collection.

Legend: `P` = paper, `B` = blog/docs, `F` = framework code, `S` = system architecture, `R` = model report, `M` = benchmark/metrics, `L` = lab/project summary.

---

## 0. Autoregressive Generation Foundations

- [x] **P** `classics/attention-is-all-you-need.md` — Transformer decoder, causal self-attention, KV reuse foundation
- [x] **P** `classics/language-models-are-unsupervised-multitask-learners.md` — GPT-2 autoregressive generation and nucleus sampling adoption
- [x] **P** `classics/gpt-3-language-models-are-few-shot-learners.md` — in-context prompting and generation at scale
- [x] **P** `classics/neural-text-degeneration.md` — Holtzman 2019 nucleus sampling; why greedy/beam can degenerate
- [x] **P** `classics/beam-search.md` — beam search mechanics, length penalty, why it is less common for chat
- [x] **B** `blogs/hf-generation-strategies.md` — Hugging Face generation guide; greedy, beam, top-k, top-p, temperature, penalties
- [x] **B** `blogs/openai-streaming-and-token-usage.md` — streaming chunks, finish reasons, token accounting, API-level view
- [x] **P** `papers/structured-generation-constrained-decoding.md` — constrained decoding / JSON grammar / regex and finite-state masking

## 1. Transformer Inference Math and Memory

- [x] **P** `classics/multi-query-attention.md` — Shazeer 2019; one KV head for faster incremental decoding
- [x] **P** `classics/grouped-query-attention.md` — Ainslie 2023; GQA as compromise between MHA quality and MQA bandwidth
- [x] **P** `classics/rope.md` — RoPE positional encoding and inference context extension consequences
- [x] **P** `classics/alibi.md` — ALiBi inference extrapolation baseline
- [x] **P** `classics/attention-complexity.md` — quadratic attention cost and prefill/decode asymmetry
- [x] **P** `classics/kv-cache-memory-formula.md` — formula card: layers × heads × head_dim × tokens × dtype × K/V
- [x] **P** `classics/prefill-vs-decode.md` — compute-bound prefill vs memory-bandwidth-bound decode
- [x] **P** `classics/batching-for-inference.md` — static batching vs dynamic batching vs continuous batching

## 2. KV Cache and Memory Management

- [x] **P** `papers/pagedattention.md` — vLLM PagedAttention; virtual-memory-style KV blocks
- [x] **P** `papers/cachegen.md` — KV cache compression/offload for long-context serving
- [x] **P** `papers/attention-sinks.md` — streaming LLM attention sinks and cache windowing
- [x] **P** `papers/h2o.md` — heavy-hitter oracle KV eviction
- [x] **P** `papers/snapkv.md` — SnapKV observation-window compression
- [x] **P** `papers/quest-kv.md` — query-aware sparsity for long-context inference
- [x] **P** `papers/infllm.md` — long-context inference with external memory / cached blocks
- [x] **F** `frameworks/vllm-kv-cache-manager.md` — vLLM block manager, paged KV cache, prefix caching
- [x] **F** `frameworks/sglang-radixattention.md` — SGLang RadixAttention and automatic prefix caching
- [x] **F** `frameworks/tensorrt-llm-paged-kv.md` — TensorRT-LLM paged KV cache and inflight batching
- [x] **B** `blogs/kv-cache-explained.md` — practitioner reference for KV cache mechanics

## 3. Scheduling and Serving Systems

- [x] **P** `papers/orca.md` — iteration-level scheduling and selective batching
- [x] **P** `papers/continuous-batching.md` — dynamic / continuous batching for token-by-token decode
- [x] **P** `papers/sarathi-serve.md` — chunked prefill to reduce decode stalls
- [x] **P** `papers/distserve.md` — disaggregated prefill and decode serving
- [x] **P** `papers/splitwise.md` — split prefill/decode placement across hardware
- [x] **P** `papers/mooncake.md` — KV cache centric disaggregated serving architecture
- [x] **P** `papers/vtc.md` — fair scheduling / virtual token counter for LLM serving
- [x] **P** `papers/niyama.md` — QoS-driven scheduling for mixed interactive/batch LLM workloads
- [x] **P** `papers/serving-optimization-foundations-2026.md` — 2026 position paper: serving needs mathematical optimization beyond FIFO/LRU heuristics
- [x] **P** `papers/llm-serving-survey.md` — recent survey of LLM serving systems and bottlenecks
- [x] **S** `systems/prefill-decode-disaggregation.md` — synthesis page for P/D disaggregation patterns
- [x] **S** `systems/admission-control-goodput.md` — SLO-aware admission control and goodput

## 4. Kernels and Runtime Optimizations

- [x] **P** `papers/flashattention.md` — IO-aware exact attention kernel
- [x] **P** `papers/flashattention-2.md` — better parallelism and work partitioning
- [x] **P** `papers/flashattention-3.md` — Hopper-era asynchronous / FP8 attention
- [x] **P** `papers/flashdecoding.md` — efficient attention for long-context decoding
- [x] **P** `papers/flashinfer.md` — LLM serving kernel library for paged attention and decoding
- [x] **P** `papers/xformers-memory-efficient-attention.md` — memory-efficient attention implementations
- [x] **P** `papers/cuda-graphs-inference.md` — CUDA graphs for reducing launch overhead
- [x] **P** `papers/tensor-parallel-inference.md` — tensor parallelism, all-reduce, and inference sharding
- [x] **P** `papers/pipeline-parallel-inference.md` — pipeline parallel inference and microbatching
- [x] **P** `papers/expert-parallel-inference.md` — MoE expert parallel inference and routing

## 5. Decoding Algorithms and Acceleration

- [x] **P** `papers/speculative-decoding.md` — Leviathan 2023 draft-and-verify algorithm
- [x] **P** `papers/fast-inference-from-transformers-via-speculative-decoding.md` — Chen et al. speculative sampling
- [x] **P** `papers/medusa.md` — multi-head self-draft decoding
- [x] **P** `papers/eagle.md` — feature-level draft model for speculative decoding
- [x] **P** `papers/eagle-2.md` — dynamic draft trees
- [x] **P** `papers/lookahead-decoding.md` — parallel n-gram / Jacobi-style decoding
- [x] **P** `papers/prompt-lookup-decoding.md` — prompt n-gram speculation
- [x] **P** `papers/self-speculative-decoding.md` — layer-skipping/self-drafting methods
- [x] **P** `papers/multi-token-prediction-inference.md` — multi-token prediction heads for faster decode
- [x] **B** `blogs/hf-assisted-generation.md` — Hugging Face assisted generation implementation

## 6. Frameworks and Code-Level References

- [x] **F** `frameworks/vllm.md` — vLLM architecture, PagedAttention, continuous batching, OpenAI-compatible server
- [x] **F** `frameworks/vllm-scheduler.md` — scheduler, prefill/decode handling, chunked prefill, priority/preemption
- [x] **F** `frameworks/vllm-structured-output.md` — guided decoding / grammar / JSON backend
- [x] **F** `frameworks/sglang.md` — SGLang runtime architecture and API
- [x] **F** `frameworks/sglang-scheduler.md` — scheduler, RadixAttention, prefix cache, disaggregation
- [x] **S** `systems/sglang-hicache.md` — SGLang hierarchical KV cache across GPU, CPU, and distributed storage
- [x] **F** `frameworks/sglang-structured-output.md` — constrained decoding and structured output stack
- [x] **F** `frameworks/tensorrt-llm.md` — NVIDIA TensorRT-LLM runtime, engines, in-flight batching
- [x] **F** `frameworks/hf-tgi.md` — Hugging Face Text Generation Inference router/server
- [x] **F** `frameworks/llama-cpp-server.md` — llama.cpp server, KV cache, slots, batching
- [x] **F** `frameworks/lightllm.md` — LightLLM serving architecture
- [x] **F** `frameworks/lmdeploy.md` — LMDeploy TurboMind and serving stack
- [x] **F** `frameworks/deepspeed-fastgen.md` — DeepSpeed-FastGen / MII serving stack

## 7. Benchmarks and Metrics

- [x] **M** `benchmarks/ttft-tpot-itl.md` — latency metric definitions and traps
- [x] **M** `benchmarks/sharegpt-workload.md` — ShareGPT request-length distribution for serving tests
- [x] **M** `benchmarks/vllm-benchmarks.md` — vLLM benchmark scripts and methodology
- [x] **M** `benchmarks/sglang-benchmarks.md` — SGLang benchmark scripts and methodology
- [x] **M** `benchmarks/genai-perf.md` — NVIDIA GenAI-Perf
- [x] **M** `benchmarks/llmperf.md` — Anyscale LLMPerf
- [x] **M** `benchmarks/mlperf-inference-llm.md` — MLPerf Inference LLM scenarios
- [x] **M** `benchmarks/helm-inference.md` — HELM / serving-oriented evaluation dimensions
- [x] **M** `benchmarks/goodput-slo.md` — goodput under latency SLOs

## 8. Production Reports and Model-Specific Inference Notes

- [x] **R** `model-reports/llama-3-inference.md` — Llama 3 GQA, long context, tokenizer, deployment notes
- [x] **R** `model-reports/qwen-3-inference.md` — Qwen 3 context lengths, GQA/MoE, serving implications
- [x] **R** `model-reports/deepseek-v3-inference.md` — DeepSeek V3 MoE, MLA, FP8 deployment details
- [x] **R** `model-reports/deepseek-r1-inference.md` — R1 long-CoT generation, sampling and serving implications
- [x] **R** `model-reports/gpt-oss-inference.md` — GPT-OSS MXFP4 MoE serving, context, tool/CoT settings
- [x] **R** `model-reports/mixtral-inference.md` — sparse MoE routing and expert parallel serving
- [x] **R** `model-reports/gemma-inference.md` — Gemma/Gemma 3 inference and local deployment notes
- [x] **R** `model-reports/phi-inference.md` — Phi family small-model local inference

## 9. Blogs, Docs, and Practitioner Guides

- [x] **B** `blogs/vllm-docs.md` — vLLM docs: serving, engines, scheduler, prefix caching
- [x] **B** `blogs/vllm-disaggregated-prefill-2026.md` — vLLM disaggregated prefilling docs + MORI-IO single-node results
- [x] **B** `blogs/vllm-kv-offloading-connector.md` — vLLM asynchronous KV offloading connector and CPU cache path
- [x] **B** `blogs/sglang-docs.md` — SGLang docs: frontend language, runtime, RadixAttention
- [x] **B** `blogs/tensorrt-llm-docs.md` — TensorRT-LLM docs and best-practice guide
- [x] **B** `blogs/hf-llm-inference-optimization.md` — HF inference optimization guide
- [x] **B** `blogs/bentoml-llm-inference-handbook.md` — production inference overview
- [x] **B** `blogs/modal-vllm-guide.md` — practical vLLM deployment guide
- [x] **B** `blogs/anyscale-llm-serving.md` — LLM serving metric and scaling discussions
- [x] **B** `blogs/lmsys-serving.md` — LMSYS/FastChat serving lessons

## 10. Labs and Projects

- [x] **L** `labs/vllm-project.md` — UC Berkeley Sky Computing Lab / vLLM project
- [x] **L** `labs/sglang-project.md` — SGLang project and maintainers
- [x] **L** `labs/nvidia-inference.md` — TensorRT-LLM, Triton, GenAI-Perf, Dynamo, NIM
- [x] **L** `labs/huggingface-inference.md` — TGI, Transformers generation, Text Generation Router
- [x] **L** `labs/llama-cpp.md` — llama.cpp local inference ecosystem
- [x] **L** `labs/lmsys-fastchat.md` — FastChat / Chatbot Arena serving lineage
- [x] **L** `labs/microsoft-deepspeed.md` — DeepSpeed inference and FastGen

---

## Gap Log

- Initial collection started 2026-05-21 with parallel agents for fundamentals, KV/scheduling, frameworks, decoding acceleration, and benchmarks/production reports.
