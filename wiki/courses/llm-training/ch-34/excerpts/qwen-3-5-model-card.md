---
chapter: ch-34
course: llm-training
phase: read
excerpt_of: "Qwen/Qwen3.5-397B-A17B model card (Hugging Face)"
source_url: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
created_at: "2026-09-15"
---

# Excerpt: Qwen3.5-397B-A17B model card — agentic RL claim and context-folding evaluation setting

The library card [[qwen-3-5]] has not been verified; its statements that Qwen3.5 inherits the Qwen3 four-stage pipeline,
uses GRPO, and has a "19× decoding speedup" were not found in the model card. This excerpt records what the official
model card states. Source type: official model card (Qwen Team); citation block dated February 2026. Read on 2026-09-15.

## Model overview
- 397B total and 17B activated parameters; 60 layers with layout "15 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated
  Attention -> MoE))"; 512 experts with 10 routed + 1 shared activated ("Model Overview").
- "Context Length: 262,144 natively and extensible up to 1,010,000 tokens." The hosted Qwen3.5-Plus uses 1M context by
  default ("Model Overview"; introduction).
- "Qwen3.5 models operate in thinking mode by default"; thinking is disabled with
  `"chat_template_kwargs": {"enable_thinking": False}` ("Quickstart"; "Instruct (or Non-Thinking) Mode"). One released
  checkpoint therefore serves both modes.
- The card advises "maintaining a context length of at least 128K tokens to preserve thinking capabilities"
  ("Serving Qwen3.5").

## Post-training statements ("Qwen3.5 Highlights")
- "Scalable RL Generalization: Reinforcement learning scaled across million-agent environments with progressively
  complex task distributions for robust real-world adaptability."
- "asynchronous RL frameworks supporting massive-scale agent scaffolds and environment orchestration."
- No algorithm, reward, environment list, step count, or ablation is given.

## Agentic evaluation settings (benchmark notes)
- "Search Agent: most search agents built on our model adopt a simple context-folding strategy(256k): once the
  cumulative Tool Response length reaches a preset threshold, earlier Tool Responses are pruned from the history to
  keep the context within limits."
- "BrowseComp: we tested two strategies, simple context-folding achieved a score of 69.0, while using the same
  discard-all strategy as DeepSeek-V3.2 and Kimi K2.5 achieved 78.6."
- "WideSearch: we use a 256k context window without any context management."
- Selected scores for Qwen3.5-397B-A17B ("Language" table): BFCL-V4 72.9, TAU2-Bench 86.7, SWE-bench Verified 76.4,
  Terminal Bench 2 52.5, BrowseComp 69.0/78.6, WideSearch 74.0.

## Verification
- Read on 2026-09-15 from the cached model card huggingface.co/Qwen/Qwen3.5-397B-A17B (main).
