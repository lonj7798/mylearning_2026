<!-- scope: HELM-style evaluation dimensions for inference efficiency and latency
     see-also: ttft-tpot-itl, llmperf
-->

# HELM Inference
- **Core Insight:** Model evaluation should include efficiency and serving behavior alongside accuracy, robustness, and fairness.
- **Guideline:** Treat HELM-style efficiency metrics as complementary to systems benchmarks: they contextualize cost and latency but do not replace load testing.
- **Authors:** Stanford CRFM HELM team
- **Year:** 2022-2026
- **URL:** https://crfm.stanford.edu/helm/
- **Relevant topics:** holistic evaluation, latency, efficiency, model comparison, benchmark reporting

## Abstract
HELM is a holistic evaluation framework for language models. Although best known for task evaluation, its reporting philosophy includes multiple dimensions such as accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency. For inference coursework, HELM is useful as an example of benchmark reporting that keeps performance and quality dimensions separate.

## Key Contributions
- Encourages multi-metric reporting instead of single leaderboard scores.
- Includes efficiency/cost/latency dimensions where available.
- Provides a structured model-evaluation framework that can be extended with serving metrics.
- Reinforces that inference speed is meaningful only with task and quality context.

## Key Figures/Tables to Study
- HELM leaderboard/report cards: multidimensional model summaries.
- HELM methodology docs: scenario and metric separation.

## Technical Details
HELM is not primarily an LLM serving load generator. It is best used to teach evaluation design: define scenarios, instances, metrics, adapters, and reporting slices. Inference systems benchmarks should borrow this discipline by pairing latency and throughput with quality, prompt distribution, and decoding settings.

## Connections
- [[llmperf]] and [[genai-perf]] provide the serving-load mechanics that HELM does not focus on.
- [[ttft-tpot-itl]] gives concrete streaming latency dimensions.
