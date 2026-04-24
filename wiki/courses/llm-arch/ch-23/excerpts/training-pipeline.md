# Excerpt: Qwen 3 Training Pipeline — From Pre-Training to Dual-Mode

<!-- source: [[qwen-3|report]] -->

## Pre-Training: Three Stages

The Qwen 3 pre-training pipeline processes 36 trillion tokens across three stages, each with a distinct purpose and data composition.

### Stage 1: General Pre-Training (~30T tokens, 4K context)

The bulk of training uses standard web-scale data: multilingual text across 119 languages and dialects, code, books, and general web content. The 4K sequence length keeps self-attention cost manageable. This stage builds the model's broad knowledge base — language understanding, world knowledge, basic reasoning patterns.

The expansion from 29 to 119 languages (compared to Qwen 2.5) requires careful data curation to prevent high-resource languages from dominating the training signal. The byte-level BPE tokenizer with 151,669 tokens provides coverage across all target languages without unknown-token fallbacks.

### Stage 2: Reasoning Enhancement (~5T tokens, 4K context)

A curated subset of high-quality STEM, coding, and reasoning data. This stage is analogous to "continued pre-training" or "domain adaptation" — the model distribution is shifted toward analytical content. The 5T token budget is substantial (roughly 14% of total pre-training compute), indicating the team's investment in reasoning as a pre-training objective, not just a post-training fine-tuning target.

### Stage 3: Long Context Extension (~100s of billions of tokens, 32K context)

Position embedding extension via RoPE interpolation, trained at 32K sequence length. The models ultimately support 128K context through further interpolation at inference time. Training at 32K rather than 128K is a compute optimization — self-attention cost scales quadratically with sequence length, so training at 32K uses 16x less attention compute per token than training at 128K.

## Post-Training: Four Stages Building Dual-Mode Capability

### Stage 1: Long-CoT Cold Start (SFT)

Supervised fine-tuning on curated chain-of-thought examples. This is *format teaching* — the model learns what extended reasoning looks like:

- Multi-step problem decomposition
- Intermediate result verification
- Backtracking when a reasoning path fails
- Structured conclusion from accumulated evidence

The data consists of carefully constructed reasoning traces, not raw model outputs. Quality matters more than quantity at this stage.

### Stage 2: Reasoning RL (GRPO, 3,995 query-verifier pairs)

Group Relative Policy Optimization uses deterministic verifiers (compilers, math checkers) as reward signals. The process:

1. Given a query, generate K candidate responses
2. Score each response with the verifier (binary: correct/incorrect for math, passes/fails for code)
3. Compute advantage relative to the group mean
4. Update policy to increase probability of high-scoring responses

The extraordinary data efficiency (3,995 examples) confirms that RL is not teaching new knowledge. The 36T pre-training tokens already provided the knowledge. RL reshapes the model's *strategy* for deploying that knowledge — which reasoning chains lead to correct answers, which approaches to try first, when to backtrack.

### Stage 3: Thinking Mode Fusion

The critical integration stage. Training data includes both:
- **Thinking examples:** Queries with `<think>...</think>` reasoning traces preceding the answer
- **Non-thinking examples:** The same or similar queries with direct answers, no reasoning trace

The model must learn two distinct behaviors and a clean switching mechanism:
- When `<think>` is present in the prompt context: generate extended reasoning
- When `<think>` is absent: generate direct answers
- Never mix modes — no partial reasoning in non-thinking mode, no skipped reasoning in thinking mode

Mode confusion is the primary failure mode. Without careful data curation, the model may hedge — generating brief reasoning traces in non-thinking mode, or truncating reasoning in thinking mode. The Qwen 3 team's approach solves this through explicit mode-labeled training data, though the exact data composition is not disclosed.

### Stage 4: General Domain RL

Final RL pass with broader reward signals (not just math/code verifiers). This prevents "reasoning mode overfitting" — the risk that stages 1-3 improve reasoning at the expense of non-reasoning capabilities. The general RL stage recovers performance on:

- Summarization and paraphrasing
- Translation across 119 languages
- Creative writing
- Factual question answering
- Instruction following

## Distillation Path (Alternative for Small Models)

For models 0.6B through 8B, the full four-stage pipeline is replaced by:

1. Generate high-quality outputs from the flagship model (235B-A22B)
2. Fine-tune the small model on these outputs (standard SFT)
3. Cost: 1/10 of the full four-stage pipeline GPU hours

Result: distilled models achieve **superior** pass@1 and pass@64 compared to equivalently-sized models trained through the full pipeline. This confirms the scale-dependent finding: RL exploration requires large model capacity; small models learn more efficiently through imitation.

The scale boundary for this crossover appears to be around 10-14B parameters. Below this, distillation dominates on both cost and quality. Above it, the full RL pipeline becomes competitive because the model has sufficient capacity to discover novel reasoning strategies through exploration — strategies that may not be present in the teacher's outputs.
