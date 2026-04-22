<!-- scope: Gemini 2.5 post-training — Deep Think / Deep Research RL innovations
     deps: []
     see-also: [[qwen-3]], [[grok-4-1]]
-->

# Gemini 2.5 (Deep Think / Deep Research)
- **Core Insight:** Combining RL + parallel multi-sampling at inference ("Deep Think") is a post-training-level commitment, not just an inference trick — the RL recipe is co-designed with sample-then-refine-then-combine inference.
- **Guideline:** When training reasoning models, shape the RL reward so the model's single-sample chain-of-thought is robust to later aggregation across multiple samples.

- **Authors / Lab:** Google DeepMind
- **Year:** 2025 (Gemini 2.5 Pro March 2025; Deep Think rollout Aug 2025)
- **URL:** https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-model-thinking-updates-march-2025/ — https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
- **Relevant topics:** thinking models, Deep Think multi-sample aggregation, Deep Research agentic RL, SFT / RM / RL data-quality focus, multi-step tool-use RL environments

## Abstract
Gemini 2.5 is Google's first generation marketed explicitly as "thinking models" — models that reason before responding. The 2.5 Pro technical report states post-training was the primary driver of the generational jump, built on improved data quality across SFT, Reward Modeling, and RL stages, plus new "algorithmic changes to the RL process" for stability during longer training. Deep Think (rolled out Aug 2025) combines RL with inference-time multi-sampling, refinement, and combination. Deep Research extends the RL stack to multi-step agentic tool-use environments.

## Key Contributions
- **"Thinking by default"** — unified native thinking integrated into post-training, not bolted on.
- **Data-quality co-improvement across SFT, RM, RL** stages as the explicit recipe focus.
- **Longer-training-stability algorithmic changes** to RL — unspecified in blog, but explicitly called out as a delta from prior Gemini.
- **Multi-step / tool-use RL environments** — RL now learns from environments with actions and tool calls, not only reasoning-reward completions.
- **Deep Think inference strategy** — sample many candidate thoughts, refine, combine. Designed jointly with post-training.
- **Deep Research** (May 2025) — agentic long-horizon research workflows, also RL-trained.

## Post-training pipeline
- **SFT data:** described only as "significantly improved data quality" — no sizes disclosed.
- **Preference / RL algorithm:** not named. Google has historically not named its RL algorithm publicly for Gemini. Blog mentions "algorithmic changes to the RL process" without specifying PPO/GRPO/DPO.
- **Reward model:** RM stage mentioned as one of three axes of data-quality focus — architecture and training recipe not disclosed.
- **KL / entropy handling:** not disclosed.
- **Rollout scale:** not disclosed.
- **Hyperparameters:** not disclosed.
- **Verifiable rewards:** multi-step / tool-use environments imply verifiable components (tool-call success, execution results) mixed with RM-scored completions.
- **Self-improvement / iterative:** Deep Research is agentic and long-horizon — implies iterative exploration + reward during training. Deep Think's inference-time self-combination is not training-time self-play but co-designed with it.

## Innovations vs predecessors
Changes from **Gemini 1.5 / 2.0 → 2.5**:
- First Gemini generation with thinking as a trained-in default.
- Deep Think = new inference strategy coupled with matching RL training.
- Deep Research extends RL to multi-step agentic environments.
- "Algorithmic changes for longer-training stability" — unnamed but called out, consistent with industry-wide entropy-collapse mitigations.

## Key Figures/Tables to Study
- 2.5 Pro technical report: benchmark deltas vs 2.0 — quantifies the post-training gain.
- Deep Think pairwise-preference plots on reasoning benchmarks.
- Deep Research agentic-task breakdown — where tool-use RL pays off.

## Connections
- [[qwen-3]] — both 2025 "hybrid thinking" releases; Qwen discloses algorithms, Google does not.
- [[grok-4-1]] — shared reasoning-model-as-RM ethos.
- [[kimi-k1-5]] — Deep Think sample-then-combine resembles K1.5's prioritized-sampling + shortest-rejection loop, but applied at inference.

## Gaps / what the report does NOT disclose
Google's Gemini blogs + tech report are exceptionally thin on post-training specifics. Not disclosed: RL algorithm name, RM architecture, KL β, LR, batch size, clip ε, group size, rollouts per prompt, RL step count, SFT data size, number/type of tool-use environments, Deep Think sampling count, Deep Research RL curriculum, multi-step reward shape. Consumer-facing blog + glossy report are the only public surfaces — no arXiv paper for 2.5 Pro's post-training.
