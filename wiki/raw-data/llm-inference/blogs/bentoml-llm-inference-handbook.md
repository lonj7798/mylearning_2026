<!-- scope: BentoML handbook-style guidance for LLM inference serving
     see-also: vllm-project, huggingface-inference
-->

# BentoML LLM Inference Handbook
- **Core Insight:** BentoML's LLM inference material packages serving concerns into deployment patterns: runtime choice, batching, autoscaling, streaming, and observability.
- **Guideline:** Use it as practitioner context for production packaging, not as the canonical source for low-level kernels.
- **Authors:** BentoML
- **Year:** 2023-2026
- **URL:** https://bentoml.com/llm/
- **Relevant topics:** deployment, serving API, batching, streaming, observability, vLLM integration

## Abstract
BentoML publishes guides and handbook material on deploying LLM inference services. The material focuses on packaging models into services, choosing runtimes such as vLLM or Transformers, exposing APIs, scaling deployments, and operating streaming inference workloads.

## Key Contributions
- Connects model-runtime choices to deployable service patterns.
- Covers practical API, container, and scaling concerns.
- Discusses streaming and batching from an application-serving perspective.
- Shows how specialized engines can sit inside a broader service framework.

## Key Figures/Tables to Study
- BentoML LLM deployment guides: service structure and runtime integration.
- vLLM integration examples: separating engine configuration from API packaging.

## Technical Details
BentoML is not itself the transformer kernel engine. It is useful for teaching the layer above inference engines: packaging, routing, deployment, autoscaling, and observability. Benchmark claims should be traced back to the underlying runtime and reproduced with explicit model and hardware settings.

## Connections
- [[vllm-project]] supplies one common backend.
- [[huggingface-inference]] supplies model packaging and TGI/Transformers paths.
