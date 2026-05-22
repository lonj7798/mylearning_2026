<!-- scope: Modal guide for deploying vLLM-backed LLM services
     see-also: vllm-docs, vllm-project
-->

# Modal vLLM Guide
- **Core Insight:** Modal's vLLM guides show the cloud-application side of serving: GPU selection, container image, model download, cold start, and endpoint exposure.
- **Guideline:** Use Modal material to teach deployment mechanics around vLLM, while using vLLM docs for engine semantics.
- **Authors:** Modal
- **Year:** 2023-2026
- **URL:** https://modal.com/docs/examples/vllm_inference
- **Relevant topics:** vLLM deployment, serverless GPUs, model caching, endpoint serving

## Abstract
Modal's vLLM examples demonstrate how to deploy an LLM inference endpoint using Modal infrastructure and vLLM. The guide covers image setup, GPU configuration, model loading/caching, and exposing an API endpoint.

## Key Contributions
- Shows a concise cloud deployment path for vLLM.
- Highlights operational issues such as model download time and GPU choice.
- Connects application endpoint code to an underlying serving engine.
- Provides a practical example for experimentation and demos.

## Key Figures/Tables to Study
- Modal example code: image, GPU, volume/model cache, and endpoint function.
- Deployment notes: cold-start and hardware selection implications.

## Technical Details
Serverless or on-demand GPU deployments must account for model weight download, container startup, cache warming, and concurrency limits. These factors affect TTFT and availability but are outside engine-only benchmarks. For steady-state measurements, separate cold-start latency from warmed endpoint serving latency.

## Connections
- [[vllm-docs]] documents the engine used inside the guide.
- [[goodput-slo]] explains production capacity measurement after deployment.
