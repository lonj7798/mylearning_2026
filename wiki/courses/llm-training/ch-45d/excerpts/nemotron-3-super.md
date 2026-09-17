---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: NVIDIA Nemotron 3 Super technical report, 2026-04-03 (no library card at the time of writing; chapter-local verified extract)
source_url: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf
created_at: "2026-09-15"
---

# Excerpt: Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

- **Authors:** NVIDIA
- **Year:** 2026 (report dated 2026-04-03; also arXiv:2604.12374)
- **Source type:** official technical report; datasets and checkpoints released on HuggingFace
- **Used in:** [[read]] §1, §2, §3, §4, §5, §7, Recipe, Generalization lens

## Scale and pre-training (Abstract, §1)
- 120B total / 12B active hybrid Mamba-Attention MoE with LatentMoE and MTP layers; pre-trained on 25T tokens in two phases — "The first phase accounted for 80% of pre-training (20 trillion tokens) and focused on diversity and broad coverage, while the second phase accounted for 20% of pre-training (5 trillion tokens) and focused on high-quality data and benchmark accuracy." Final context length up to 1M.
- §2.3 lists the pre-training data, including synthetic code concepts, algorithmic, economics, formal-logic and multiple-choice sets. No agentic trajectory corpus is listed in the pre-training data section.

## Post-training pipeline (§3, Figure 12)
- SFT ("7M Samples, 80B Tokens") → RLVR in three rounds (Round 1: 25 environment types; Round 2: 30 environment types with low effort; Round 3: 26 environment types, agentic-focused) → SWE RL (20B tokens) → RLHF (18B tokens) → MTP healing. Figure 12 also prints "37 environment types" and "Up to 4,000 environment instances per batch".
- The pipeline has no specialist-distillation stage; one model passes through all stages.

## SFT agentic data (§3.1.1, §3.1.2)
- Software engineering: issues and containerized environments from SWE-Gym, R2E-Gym and SWE-rebench; R2E-Gym problem statements regenerated with Qwen3-Coder-480B-A35B-Instruct; trajectories distilled from the OpenHands harness with the same teacher.
- Agentic CLI: NeMo Data Designer generates "approximately 20k queries derived from a taxonomy of 24 distinct actions"; GPT-OSS-120B filters them, leaving "roughly 15k tasks centered on direct solution synthesis", each paired with an AGENTS.md-style specification. Augmented with "roughly 3000 questions from SWE tasks" and "10k web development tasks using a taxonomy of 100 fine-grained tasks".
- CLI interaction traces are recorded from Qwen3-Coder-480B and MiniMax M2.5 across Codex, OpenCode, Qwen Code CLI and Stirrup, then normalized into OpenAI message format.
- Terminal use: 84,864 samples following the Terminal-Task-Gen methodology (68,924 synthetic, 8,125 Nemotron-Cascade-Math, 7,815 Nemotron-Cascade-Code), generated with DeepSeek-V3.2 inside Dockerized environments under the Terminus 2 framework.
- Conversational tool use: a six-stage synthetic pipeline (domain generation, policy and tool generation, scenario generation, "simulate 16 customer service interactions" per policy–scenario pair, LM-as-a-Judge verification at outcome and process level, difficulty filtering that drops all-success and all-failure scenarios) "yielding 279,116 conversations across 838 domains. This represents a substantial scale-up over Nemotron 3 Nano, which used 15,588 conversations spanning 5 domains."
- General-purpose tool use: three-role simulation (User-LLM, Assistant-LLM, Tool-LLM) over tool sets from ToolEyes, API-Bank, UltraTools, AutoTools, xLAM, Glaive-Function-Calling-v2 and Toucan-1.5M, scaled with DeepSeek-V3.2 and GLM-4.7 "to create a dataset of 1.5M diverse tool-calling trajectories".
- Search: knowledge-graph walks turned into obfuscated multi-hop questions, solved by MiniMax-M2 via the Tavily MCP search tool, "preserving a full Thought–Action–Observation loop across an average of 12 tool calls per trajectory".

## SFT blend (§3.1.2)
- "We train on over 7M total samples. In stage 2, we use 85% of the stage 1 blend and augments it with 256K and 512K-token long-context data. Compared to Nemotron 3 Nano, we significantly increased the volume and diversity of agentic tasks, and allocate it a much larger proportion of our blend." Exact percentages appear only in Figure 16.

## RL stages (§3.2)
- Stage 1 (multi-environment RLVR) is "the primary training stage": "Training in a unified mixture keeps each RL update informed by the complete environment distribution and helps prevent regressions on individual tasks over the course of training."
- Generality result, stated without a table: "We find that training on all environments simultaneously yields stable gains, whereas single-environment training leads to severe regressions on other benchmarks."
- 21 environments and 37 RL datasets covering math, code, STEM, instruction following, safety (over-refusal and jailbreak robustness), long context, agentic tool use (conversational and terminal), and Reasoning Gym. Prompts the SFT model always answers correctly are filtered out, then sorted by a difficulty curriculum.
- Stage 2 (SWE-RL) is separate "because SWE rollouts are substantially slower to generate and typically require longer context lengths, creating a throughput bottleneck when co-trained with shorter-horizon environments". Each rollout runs an OpenHands loop in an Apptainer container with a binary test-based reward; OpenCode and Codex agent classes are implemented inside OpenHands so that "This multi-harness training improves the model's generalization and performance across all target harnesses at inference time."
- Stage 3 (RLHF) uses a principle-following GenRM initialized from Qwen3-235B-A22B-Thinking-2507, trained on HelpSteer 3, commercially friendly lmarena-140k subsets and newer preference data; the GenRM is used throughout multi-environment RL and again in a separate RLHF stage.
- Stage 4: MTP healing trains only the MTP heads with an NLL loss on RLVR-prompt generations.

## RL algorithm and settings (§3.2.4)
- Asynchronous GRPO with decoupled training and inference; "we restrict the inference workers to be at most one step behind the latest model version"; the KV cache is not recomputed after weight updates; the importance-sampling ratio computed from training and inference log-probabilities is masked.
- "In multi-environment RLVR, we sample 256 prompts per step and generate 16 responses per prompt. We train with a batch size of 4096, which corresponds to a single gradient update per rollout. We begin training with a maximum generation length of 49K tokens and later increase it to 64K."
- PivotRL is applied to "all agentic domains: including for Agentic Programming, Search, Terminal Use, and Conversational Tool Use". It "reuses offline SFT expert trajectories during RL", focusing on turns where the policy is uncertain, with a domain-appropriate reward matching the policy action to the expert action. Stated motivation: SFT "often degrades performance outside of the target domain (OOD)", while end-to-end RL avoids that but is costly.

## Post-trained evaluation (§3.3)
- Selected Nemotron 3 Super scores: MMLU-Pro 83.73; AIME25 (no tools) 90.21; GPQA (no tools) 79.23; SWE-Bench 60.47 (OpenHands), 59.20 (OpenCode), 53.73 (Codex); Terminal Bench Core 2.0 31.00; TauBench V2 average 61.15; BrowseComp with Search 31.28; IFBench (prompt) 72.56; Arena-Hard-V2 73.88.

## Verification
- Checked on 2026-09-15 against the NVIDIA Nemotron 3 Super technical report (2026-04-03): Abstract, §1, §2.3, §3, §3.1.1-§3.1.3, §3.2.1-§3.2.5, §3.3.
- Not reported by the source: exact SFT blend percentages (figure only); RL learning rate, clip values, KL coefficient; number of RL prompts per environment; GPU hours; a controlled before/after table for the single-environment regression claim.
