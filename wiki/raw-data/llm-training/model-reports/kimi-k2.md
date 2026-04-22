<!-- scope: Kimi K2 — Moonshot AI's 1T-MoE agentic model
     deps: [[README]]
     see-also: [[deepseek-v3]], [[deepseek-r1]]
-->

# Kimi K2: Open Agentic Intelligence
- **Core Insight:** Pretraining stability at trillion-parameter scale is unlocked by Muon + QK-Clip (MuonClip); alignment for agentic behavior requires a joint RLVR + self-critique rubric-reward training stage.
- **Guideline:** For trillion-parameter MoE, swap AdamW for MuonClip and instrument attention-logit max during training; for agentic RL, supplement verifiable rewards with model-generated rubric rewards.
- **Authors:** Kimi Team (Moonshot AI)
- **Year:** 2025 (arXiv Jul 2025)
- **URL:** https://arxiv.org/abs/2507.20534
- **Relevant topics:** MuonClip optimizer, 1T-MoE pretraining, agentic data synthesis, RLVR + self-critique rubric rewards, 20K+ tools

## Abstract
Kimi K2 is a 1T-parameter MoE model with 32B activated parameters. It is pretrained for 15.5 trillion tokens with zero loss spikes using MuonClip — the Muon optimizer extended with QK-Clip to bound attention logits. Post-training is a multi-stage pipeline centered on a large-scale agentic data synthesis pipeline (20K+ tools, real and simulated environments) and a joint RL stage combining verifiable rewards (RLVR) with a self-critique rubric reward. K2 matches or beats closed frontier models on agentic benchmarks.

## Key Contributions
- MuonClip optimizer: Muon + QK-Clip that rescales Q/K weights post-update to cap attention logit magnitude, preventing logit explosion that otherwise caused divergence in mid-scale Muon runs.
- 15.5T-token pretraining at 1T parameters with no loss spikes.
- Agentic data synthesis pipeline: 20K+ tools (real + simulated), synthetic multi-turn tool-use traces.
- Joint RL stage: verifiable rewards (RLVR) + self-critique rubric rewards (the model evaluates its own outputs against a rubric the model itself produces).
- Open-weights release for research.

## Key Figures/Tables to Study
- **Loss-spike diagnostic plot:** attention-logit max reaching 1000+ in plain Muon vs bounded under MuonClip.
- **Agentic benchmark table:** K2 vs Claude 3.5 Sonnet, GPT-4o on SWE-bench, WebArena, Tau-bench.
- **Tool library breakdown:** 20K+ tools by domain.

## Technical Details — Post-Training Pipeline

### Pretraining (context for post-training)
- **Total parameters:** 1T (1 trillion); active per token 32B.
- **Pretraining tokens:** 15.5T.
- **Optimizer:** MuonClip = Muon + QK-Clip. Muon gives advanced token efficiency; QK-Clip rescales query/key projection matrices after each update so that max attention logit stays below a threshold (otherwise observed to blow past 1000). Result: zero loss spikes across full run.

### Agentic data synthesis (the SFT backbone)
- **Tool library:** 20,000+ tools covering software engineering, web navigation, data analysis, etc.
- **Environments:** both real execution sandboxes and simulated environments generate ground-truth trajectories.
- Trajectories are filtered by outcome-success verifiers before becoming SFT data.

### Joint RL stage
Kimi K2 runs a joint RL stage that combines two reward streams:
1. **RLVR (verifiable rewards)** — for tasks with checkable outcomes (math, code, tool-call correctness).
2. **Self-critique rubric reward** — the model produces both (a) a rubric appropriate for the task and (b) its own completion, then scores completions against the rubric. This extends alignment to open-ended tasks where no automated verifier exists.
- The two reward streams are combined into a single scalar reward for policy optimization.
- Specific algorithm, KL coefficient, rollout count, and LR are not exhaustively disclosed in the public report; it describes the framework rather than all hyperparameters.

### What's novel
- Self-critique as reward source avoids the drift problem of fixed reward models on open-ended tasks.
- The joint objective means a single RL stage can train both verifiable and open-ended skills simultaneously.

### Benchmark highlights
- State-of-the-art among open weights on SWE-bench Verified and Tau-bench agentic suites.
- Competitive with Claude 3.5 Sonnet on tool-use tasks.

## Connections
- [[deepseek-v3]] — parallel trillion-param MoE story; DeepSeek used FP8 + auxiliary-loss-free routing for stability, Kimi used MuonClip.
- [[deepseek-r1]] — R1's reasoning RL vs Kimi's agentic RL are both rule-reward-heavy but target different skill slices.
- [[constitutional-ai]] — self-critique rubric reward is a descendant of CAI's self-rating idea.
- [[self-rewarding-lm]] — Yuan 2024; another self-evaluation-as-reward line.
- [[tulu-3]] — RLVR component is the Tulu contribution Kimi integrates into its joint objective.
