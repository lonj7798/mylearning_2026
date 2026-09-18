<!-- scope: Mistral AI launch post "Large Enough" for Mistral Large 2 (mistral-large-2407, July 2024), with the official Hugging Face model card for Mistral-Large-Instruct-2407 as companion for printed benchmark values
     deps: none
     see-also: [[mistral-nemo]], [[pixtral-large]], [[mixtral]], [[llama-3]]
-->

# Large Enough
- **Core Insight:** Mistral Large 2 is a 123B-parameter model with a 128k context window designed for single-node inference; its pretrained version scores 84.0% on MMLU, and Mistral AI states that it was trained on "a very large proportion of code" and "a large proportion of multilingual data" (post sections "Mistral Large 2", "General performance", "Code & Reasoning", "Language diversity").
- **Guideline:** When the deployment target values short and low-cost responses, report average generation length next to alignment benchmark scores, because Mistral AI notes that lengthy responses raise scores on some benchmarks and publishes MT-Bench generation lengths beside its MT-Bench, Wild Bench, and Arena Hard results (section "Instruction following & Alignment").
- **Authors:** Mistral AI team (official post byline)
- **Year:** 2024 (published 2024-07-24)
- **URL:** https://mistral.ai/news/mistral-large-2407 ; companion model card: https://huggingface.co/mistralai/Mistral-Large-Instruct-2407
- **Source type:** official blog (companion: model/dataset card)
- **Relevant topics:** code-heavy pre-training data, multilingual data, hallucination reduction, response-length control, function calling, single-node deployment

## Summary
The post announces Mistral Large 2, available on la Plateforme as `mistral-large-2407` (version 24.07). The model has 123B parameters, a 128k context window, support for dozens of natural languages and 80+ coding languages, and is designed to run at large throughput on a single node. Instruct weights are released under the Mistral Research License, which allows research and non-commercial use; commercial self-deployment requires a Mistral Commercial License. The post describes training priorities without a recipe: a very large proportion of code, a large proportion of multilingual data, fine-tuning toward cautious answers and acknowledging missing information, succinct generations, and parallel and sequential function calling. Benchmark results in the post are images; the model card prints the corresponding numbers.

## Key Contributions
- Release of a 123B dense model with a 128k context window sized for single-node inference (post "Mistral Large 2"; model card "Model Card", "Key features").
- Stated data priorities: a very large proportion of code, following Codestral 22B and Codestral Mamba, and a large proportion of multilingual data (post "Code & Reasoning", "Language diversity").
- Stated alignment targets: fewer plausible but incorrect answers, acknowledging insufficient information, precise instruction following, long multi-turn conversations, and succinct outputs (post "Code & Reasoning", "Instruction following & Alignment").
- Training for parallel and sequential function calls and enhanced retrieval skills (post "Tool Use & Function Calling").

## Key Figures/Tables to Study
- Post figures (images): code generation benchmarks and MultiPL-E; GSM8K (8-shot) and MATH (0-shot, no CoT); MT-Bench, Wild Bench, Arena Hard; average MT-Bench generation length by model; Multilingual MMLU on the base model; one figure in the "Tool Use & Function Calling" section. The code, MultiPL-E, math, and alignment captions state that all models were benchmarked through the same evaluation pipeline, except the "paper" row in MultiPL-E.
- Model card "Metrics" tables: the printed values listed below.

## Technical Details
**Model.** 123B parameters (post "Mistral Large 2"); the model card calls it a dense LLM (model card intro). Context window 128k (post; model card "Key features"). Natural languages named: French, German, Spanish, Italian, Portuguese, Arabic, Hindi, Russian, Chinese, Japanese, Korean (post "Mistral Large 2"). Coding languages: 80+, including Python, Java, C, C++, JavaScript, and Bash (post); the model card adds Swift and Fortran (model card "Key features").

**Training signals disclosed.** Code share: "a very large proportion of code" (post "Code & Reasoning"). Multilingual share: "a large proportion of multilingual data" (post "Language diversity"). The model was fine-tuned "to be more cautious and discerning in its responses" to reduce hallucination, and trained "to acknowledge when it cannot find solutions or does not have sufficient information" (post "Code & Reasoning"). Mistral AI "spent a lot of effort to ensure that generations remain succinct" (post "Instruction following & Alignment"). The model "has undergone training to proficiently execute both parallel and sequential function calls" (post "Tool Use & Function Calling"). No stage, method, data size, or hyperparameter is given for any of these.

**Printed results.** Pretrained model: MMLU 84.0% (post "General performance"; model card "Base Pretrained Benchmarks"). Base-model Multilingual MMLU: French 82.8%, German 81.6%, Spanish 82.7%, Italian 82.7%, Dutch 80.7%, Portuguese 81.6%, Russian 79.0%, Korean 60.1%, Japanese 78.8%, Chinese 74.8% (model card "Base Pretrained Multilingual Benchmarks"). Instruct model: MT Bench 8.63, Wild Bench 56.3, Arena Hard 73.2; HumanEval 92%, HumanEval Plus 87%, MBPP Base 80%, MBPP Plus 69%; GSM8K 93%, Math Instruct 70% (0-shot, no CoT) and 71.5% (0-shot, CoT) (model card "Metrics"). The post states that the model performs on par with GPT-4o, Claude 3 Opus, and Llama 3 405B on code and reasoning (post "Code & Reasoning").

**Deployment.** Available on la Plateforme and le Chat, on Vertex AI, Azure AI Studio, Amazon Bedrock, and IBM watsonx.ai; fine-tuning on la Plateforme is offered for Mistral Large, Mistral Nemo, and Codestral (post, final sections). The model card states the model "does not have any moderation mechanisms" (model card "Limitations").

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Mistral Large 2 (mistral-large-2407) | 123B | not stated (training data) | Code share of data | "a very large proportion of code"; percentage not printed | post "Code & Reasoning" | verified 2026-09-14 (wording); not reported (percentage) | no ablation reported; code benchmark figures are images |
| Mistral Large 2 (mistral-large-2407) | 123B | not stated (training data) | Multilingual share of data | "a large proportion of multilingual data"; percentage not printed | post "Language diversity" | verified 2026-09-14 (wording); not reported (percentage) | base-model Multilingual MMLU, model card "Metrics" |
| Mistral Large 2 (mistral-large-2407) | 123B | not stated | Context window | 128k | post "Mistral Large 2"; model card "Key features" | verified 2026-09-14 | no ablation reported |
| Mistral Large 2 (mistral-large-2407) | 123B | not stated | Hallucination / abstention target | fine-tuned to be "more cautious and discerning"; trained to acknowledge insufficient information; method not stated | post "Code & Reasoning" | verified 2026-09-14 (target); not reported (method) | post links it to math benchmark results (GSM8K 8-shot, MATH 0-shot no CoT; images) |
| Mistral Large 2 (mistral-large-2407) | 123B | not stated | Function calling | trained for parallel and sequential function calls; data and method not stated | post "Tool Use & Function Calling" | verified 2026-09-14 (target); not reported (method) | figure in the same section (image) |
| Mistral Large 2 (mistral-large-2407) | 123B | not stated | Response length | effort to keep generations succinct; no length target printed | post "Instruction following & Alignment" | verified 2026-09-14 (target); not reported (value) | MT-Bench average generation length figure (image) |
| Mistral Large 2 (mistral-large-2407) | 123B | all | Tokens, mixture percentages, optimizer, LR schedule, batch size, SFT / preference / RL method, data sizes, compute | not printed | post; model card | not reported (post and model card checked; config.json is gated and was not read) | n/a |
| Mistral-Large-Instruct-2407 | 123B | eval-gate | Printed benchmark values | see "Printed results" above; evaluation settings beyond shot counts not printed | model card "Metrics" | verified 2026-09-14 | n/a |

## Findings relevant to generality, negative feedback, long context, agentic training
- **Generality.** Mistral AI states that most models are English-centric and trains on a large proportion of multilingual data; base-model Multilingual MMLU ranges from 60.1% (Korean) to 82.8% (French) (post "Language diversity"; model card "Metrics"). The post notes that lengthy responses raise scores on some benchmarks, which it treats as a reason to control length (post "Instruction following & Alignment").
- **Negative feedback.** Abstention ("acknowledge when it cannot find solutions") is a stated training target; the post does not say whether it is trained as target content, a preference signal, or another method, and reports no abstention rate (post "Code & Reasoning").
- **Long context.** 128k context and single-node inference "with long-context applications in mind" are stated; the post text reports no long-context evaluation (post "Mistral Large 2").
- **Agentic training.** Parallel and sequential function calling and retrieval are training targets; no function-calling score is printed in text (post "Tool Use & Function Calling").

## Connections
- [[mistral-nemo]]: named in the post as the other general-purpose model on la Plateforme.
- [[mixtral]]: Mixtral 8x7B and 8x22B are named as Apache models that remain available.
- [[llama-3]]: Llama 3 405B and Llama 3.1 models are the post's comparison models for code and multilingual MMLU.
- [[pixtral-large]]: its card describes a later model built on Mistral Large 2 (not stated in this post).

## Verification
- Checked on 2026-09-14 against: https://mistral.ai/news/mistral-large-2407 (post dated 2024-07-24) and https://huggingface.co/mistralai/Mistral-Large-Instruct-2407 (model card page; weight files and config.json are gated)
- Corrections to the previous card version:
  - Title "Mistral Large 2" → published post title "Large Enough"; URL normalized from `/en/news/` to `/news/mistral-large-2407`.
  - "flagship text model for long-context and enterprise use" → the post says "designed for single-node inference with long-context applications in mind" and names Mistral Nemo and Mistral Large as the two general-purpose models on la Plateforme.
  - "benchmark blocks ... and function calling" → the function-calling section contains one figure with no text caption; no function-calling score is printed in text.
  - deps `[[mistral-nemo]], [[toolformer]]` → none; neither is a prerequisite stated by the post.
- Removed as unsupported: "Why it matters" section ("product surface area"; "by mid-2024 tool use and concision were already being treated as core post-training targets in flagship models"); the old Guideline ("optimize for the capability bundle people actually deploy"); "llama-3 and qwen-2.5 are stronger if you want more public detail" (evaluative; Qwen2.5 is not mentioned in the post); "brevity was itself a training objective" (replaced by the post's wording).
- Not reported by the source: architecture details (layers, hidden size, attention type, vocabulary); pre-training token count and mixture percentages; optimizer and schedule; SFT, preference, and RL methods and data; compute; context-extension method.
