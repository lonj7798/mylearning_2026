<!-- scope: Kimi K2 technical report (arXiv:2507.20534): 1.04T-total / 32B-activated MoE, MuonClip pre-training on 15.5T tokens with rephrased data, SFT with synthetic tool-use trajectories, joint RL with verifiable rewards and a self-critique rubric reward
     deps: [[kimi-k1-5]], [[deepseek-v3]]
     see-also: [[kimi-k2-recipe]], [[kimi-k2-agentic-data]], [[rephrasing-the-web]], [[rlhf-instructgpt]], [[toolllm]]
-->

# Kimi K2: Open Agentic Intelligence
- **Core Insight:** Kimi K2, an MoE model with 1.04T total and 32B activated parameters, was pre-trained on 15.5T tokens with the MuonClip optimizer without a loss spike, and after SFT on synthetic tool-use data and a joint RL stage it reports 65.8 on SWE-bench Verified (agentic, single attempt) and 66.1 on Tau2-Bench in non-thinking evaluation (Abstract, §2.1, §2.3, Table 3).
- **Guideline:** When pre-training with Muon at scale, log the per-head maximum attention logit and bound it by rescaling query/key weights, because vanilla Muon on a 9B-activated / 53B-total MoE produced max logits above 1000, while K2 trained with QK-Clip at τ = 100 showed no loss spikes and a 0.5B-activated / 3B-total ablation with τ = 30 showed negligible loss change (§2.1, Figures 2-3, App. D).
- **Authors:** Kimi Team (Moonshot AI). App. A lists contributors alphabetically by last name: Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, et al.
- **Year:** 2025 (arXiv v1 2025-07; v2 2026-02)
- **URL:** https://arxiv.org/abs/2507.20534
- **Source type:** official technical report
- **Relevant topics:** MuonClip / QK-Clip, sparsity scaling law, knowledge and math rephrasing, long-context activation, agentic data synthesis (MCP and synthetic tools, user simulation, tool simulator), RL gym with verifiable rewards, self-critique rubric reward, budget control, PTX loss, temperature decay, partial rollout

## Abstract
Kimi K2 is a Mixture-of-Experts (MoE) LLM with 32B activated and 1T total parameters. The authors propose MuonClip, which adds a QK-clip technique to the Muon optimizer to address training instability while keeping Muon's token efficiency; with it K2 was pre-trained on 15.5T tokens with zero loss spikes. Post-training has multiple stages and features a large-scale agentic data synthesis pipeline and a joint RL stage in which the model learns from interaction with real and synthetic environments. The report states K2 is state of the art among open-source non-thinking models, with 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, 47.3 on SWE-Bench Multilingual, 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. Base and post-trained checkpoints are released (Abstract).

## Key Contributions
- MuonClip: Muon with weight decay, update-RMS matching, and per-head QK-Clip, used for the full 15.5T-token run (§2.1, Algorithm 1).
- Rephrasing pipelines for knowledge and math text to increase token utility, with a SimpleQA comparison against multi-epoch repetition (§2.2, Table 1).
- An architecture derived from DeepSeek-V3 with higher sparsity (384 experts, 8 active) and 64 attention heads, justified by a sparsity scaling law and a heads ablation (§2.3, Table 2, Figures 5-6).
- A tool-use data synthesis pipeline: tool specs → agents and rubric-based tasks → simulated multi-turn trajectories filtered by an LLM judge, plus real execution sandboxes (§3.1.1).
- A joint RL framework: a Gym-like set of verifiable-reward tasks plus a self-critique rubric reward for subjective tasks, with budget control, PTX loss, and temperature decay (§3.2).

## Key Figures/Tables to Study
- Figure 2 and App. D: max attention logits with vanilla Muon vs MuonClip; QK-Clip activity over training.
- Table 1: SimpleQA accuracy for repetition vs rephrasing. Table 2: K2 vs DeepSeek-V3 architecture.
- Figures 8-9: tool-use synthesis pipeline and t-SNE of real MCP vs synthetic tools.
- Table 3 (Kimi-K2-Instruct vs DeepSeek-V3-0324, Qwen3-235B-A22B, Claude Sonnet 4, Claude Opus 4, GPT-4.1, Gemini 2.5 Flash) and Table 4 (base model).
- App. F: core rubrics, prescriptive rubrics, and the stated limitations of the rubric framework.

## Technical Details
Recipe values with loci are in [[kimi-k2-recipe]]. This section summarizes mechanisms.

**Architecture.** 61 layers, MLA attention, hidden dimension 7168, expert hidden dimension 2048, 384 experts with 8 active per token and 1 shared expert, 64 attention heads, 1 dense layer, no expert grouping; 1.04T total and 32.6B activated parameters (§2.3, Table 2). Sparsity (total/activated experts) 48 was chosen; at equal validation loss 1.5 it needs 1.69×, 1.39×, and 1.15× fewer FLOPs than sparsity 8, 16, and 32 (§2.3, Figure 5). At 128k sequence length, 128 instead of 64 heads increases inference FLOPs by 83%, while doubling heads lowered validation loss by only 0.5% to 1.2% (§2.3, Figure 6).

**MuonClip.** The per-head max logit S^h_max is the largest pre-softmax attention score in the batch. When S^h_max > τ, the head's weights are scaled with γ_h = τ / S^h_max: for MLA, the head-specific query and key components q^C and k^C each by √γ_h, the head-specific rotary query q^R by γ_h, and the shared rotary key k^R is left unchanged (§2.1, Algorithm 1). The clip does not change the current step's forward or backward pass (§2.1). In K2, 12.7% of heads triggered QK-Clip at least once in the first 70,000 steps; after that point every head had at some point reduced S^h_max below 100, and the clip became inactive (App. D). The authors hypothesize Muon is more prone to logit explosion because its updates have full effective rank (App. E, Interpretation).

**Pre-training data.** 15.5T tokens over Web Text, Code, Mathematics, and Knowledge, with pipelines mostly following K1.5 (§2.2). Knowledge rephrasing uses style- and perspective-diverse prompts (inspired by WRAP), chunk-wise autoregressive rewriting, and fidelity verification; each corpus is rephrased at most twice (§2.2). Math documents are rewritten in a "learning-note" style following SwallowMath, and math material in other languages is translated into English (§2.2).

**SFT.** The Muon optimizer is used in post-training (§3.1). Candidate responses come from K1.5 and in-house domain-specialized expert models and are filtered by LLM or human judges (§3.1).

**Tool-use data synthesis.** 3000+ real MCP tools are fetched from GitHub and over 20,000 synthetic tools are produced by hierarchical domain evolution (§3.1.1). Thousands of agents are created from synthesized system prompts and tool combinations; each task carries a rubric with success criteria, expected tool-use patterns, and checkpoints (§3.1.1). Trajectories are generated with LLM-simulated users and a stateful tool simulator with controlled stochasticity; an LLM judge keeps only trajectories that meet the rubric (§3.1.1). Real execution sandboxes are added for coding and software engineering, with test-suite pass rates as feedback (§3.1.1). The authors describe the filtering as large-scale rejection sampling (§3.1.1).

**Verifiable rewards gym.** Math, STEM, and logic prompts are selected for coverage (tagging system) and moderate difficulty measured by the SFT model's pass@k (§3.2.1). Instruction following uses code-interpreter checks, LLM-as-judge checks, and a hack-check layer that detects claimed but unfulfilled compliance (§3.2.1). Faithfulness uses a sentence-level faithfulness judge as reward model (§3.2.1). Coding uses open-source and synthetic problems; software engineering uses GitHub issues and pull requests with unit tests in a Kubernetes sandbox supporting over 10,000 concurrent instances (§3.2.1). Safety uses an attack model, target model, and judge model with a task-specific rubric giving a binary label (§3.2.1).

**Self-critique rubric reward.** K2's critic ability is initialized in SFT from open-source and in-house preference data (§3.2.2). The K2 actor generates responses; the K2 critic ranks them by pairwise evaluation against core rubrics (App. F.1), prescriptive rubrics intended to eliminate reward hacking (App. F.2), and human-annotated rubrics for specific contexts (§3.2.2). During RL, the critic is updated with on-policy rollouts from verifiable-reward prompts (§3.2.2).

**RL algorithm.** For each problem x, K responses are sampled from π_old and the loss is E_x[(1/K) Σ_i (r(x, y_i) − r̄(x) − τ log(π_θ(y_i|x)/π_old(y_i|x)))²] (§3.2.3). r̄(x): mean reward of the K samples; τ > 0: regularization parameter (value not reported). Muon minimizes it (§3.2.3). Additions: a per-sample maximum token budget set by task type, with truncation and a penalty when exceeded; an auxiliary PTX loss on hand-selected high-quality samples; and a temperature decay schedule from exploration to exploitation (§3.2.3).

**RL infrastructure.** Training and inference engines are colocated as in K1.5; a checkpoint engine broadcasts full parameters and completes a K2 update in under 30 seconds (§3.3.1, §3.3.2). Agentic rollouts use dedicated environment services, many concurrent rollouts, partial rollout for long-tail trajectories, and a Gym-like interface (§3.3.4).

## Recipe ledger
Full table (pre-training, annealing and long-context, SFT, agentic synthesis, RL, evaluation settings): [[kimi-k2-recipe]].

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Generality.** The PTX loss is added to prevent forgetting of high-quality data and to reduce overfitting to the limited RL task set; the authors state it improves generalization across domains but give no ablation (§3.2.3). The budget control is motivated by the observation that the benefit of longer responses often does not justify their inference cost in non-reasoning domains (§3.2.3). Kimi-K2-Instruct reports MMLU 89.5, IFEval 89.8, and SimpleQA 31.0 (GPT-4.1: 42.3) (Table 3). Limitations: performance may decline on some tasks when tool use is enabled unnecessarily (§5).
- **Negative feedback.** Trajectories that fail the rubric are discarded (negative marginal value; rejection rate not reported) (§3.1.1). Over-budget responses are truncated and penalized (negative reward) (§3.2.3). The authors note that prescriptive rubrics penalize hedging and disclaimers, so the model may overstate certainty in ambiguous contexts (App. F.3).
- **Long context.** Annealing and long-context activation: 400B tokens at 4k, then 60B tokens at 32k, then YaRN to 128k (§2.5). Kimi-K2-Instruct: LongBench v2 49.1, FRAMES 77.1, MRCR 55.0, DROP 93.5 (Table 3); on FRAMES and LongBench v2 it trails DeepSeek-V3-0324 by about 2% (App. C).
- **Agentic training.** Tau2 per-domain Avg@4: retail 70.6, airline 56.5, telecom 65.8; SWE-bench Verified 51.8 agentless, 65.8 agentic single attempt, 71.6 multiple attempts (Table 3). Limitations: excessive tokens on hard reasoning or unclear tool definitions can truncate outputs or tool calls, and one-shot prompting for full software projects is weaker than an agentic coding framework (§5).
- **Distillation.** SFT responses are generated by K1.5 and domain expert models (§3.1). The report describes critic updates from verifiable-reward rollouts as distilling RLVR performance signals into the evaluation model (§3.2.2).

## Connections
- [[kimi-k1-5]] supplies the RL objective, partial rollout, colocated RL design, and most data-processing pipelines (§2.2, §3.2.3, §3.3).
- [[kimi-k2-agentic-data]] is a separate card on this report's tool-use synthesis section (§3.1.1).
- [[deepseek-v3]] is the architecture K2 compares against in Table 2.
- [[rephrasing-the-web]] (WRAP) inspired the knowledge rephrasing prompts (§2.2); [[wizardlm]] is cited for evolving synthetic tools (§3.1.1).
- [[rlhf-instructgpt]] is the cited source of the PTX loss (§3.2.3).
- [[agentinstruct]], [[self-instruct]], and [[toolllm]] are cited prior work on synthetic and tool-use data (§3.1.1).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2507.20534 (v2, 2026-02-03; v1 2025-07-28 compared, differences are reference numbering)
- Corrections to the previous card version:
  - "20K+ tools (real + simulated)" → 3000+ real MCP tools plus over 20,000 synthetic tools (§3.1.1).
  - "the model produces both (a) a rubric ... and (b) its own completion, then scores" → the K2 critic ranks actor responses by pairwise comparison against core, prescriptive, and human-annotated rubrics; the critic is refined with verifiable-reward rollouts (§3.2.2).
  - "K2 vs Claude 3.5 Sonnet, GPT-4o on SWE-bench, WebArena, Tau-bench" and "Competitive with Claude 3.5 Sonnet" → baselines are DeepSeek-V3-0324, Qwen3-235B-A22B, Claude Sonnet 4, Claude Opus 4, GPT-4.1, Gemini 2.5 Flash; benchmarks include SWE-bench Verified and Tau2-Bench; no WebArena (§4.1.1, Table 3).
  - "Trajectories are filtered by outcome-success verifiers" → an LLM judge checks trajectories against task rubrics; real sandboxes give test pass rates (§3.1.1).
  - "Specific algorithm ... not exhaustively disclosed" → the objective is given (§3.2.3); τ, K, LR, and schedules are not.
  - "logit explosion that otherwise caused divergence in mid-scale Muon runs" → the mid-scale run exceeded max logit 1000; the report says logits at this level "usually result in" loss spikes and occasional divergence (§2.1).
- Removed as unsupported by the source: "alignment for agentic behavior requires a joint RLVR + self-critique ... stage"; "The two reward streams are combined into a single scalar reward"; "Self-critique as reward source avoids the drift problem of fixed reward models"; connection claims that the RLVR component comes from [[tulu-3]] and that the rubric reward descends from [[constitutional-ai]] or [[self-rewarding-lm]] (none is cited by the report).
- Not reported by the source: pre-training mixture percentages; Muon momentum; GPU hours; YaRN parameters; SFT dataset size and hyperparameters; count of synthetic trajectories beyond "tens of thousands of diverse and high-quality training examples" (§3.1.1); RL τ, K, LR, steps, budgets per task type, PTX weight, temperature schedule; any injection of agentic trajectories into pre-training data.
