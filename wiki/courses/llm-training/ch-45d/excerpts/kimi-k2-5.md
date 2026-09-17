---
chapter: ch-45d
course: llm-training
phase: read
excerpt_of: arXiv:2602.02276v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2602.02276
created_at: "2026-09-15"
---

# Excerpt: Kimi K2.5: Visual Agentic Intelligence

- **Authors:** Kimi Team (Moonshot AI)
- **Year:** 2026 (arXiv v2 2026-08-07; v1 2026-02)
- **Source type:** official technical report (checkpoint: https://huggingface.co/moonshotai/Kimi-K2.5)
- **Used in:** [[read]] §1, §2, §3, §4, §5, §6, §7, Recipe, Generalization lens

## Stage placement (§4.1, §4.3, §4.4)
- Built on the Kimi K2 language-model checkpoint (1.04T total / 32B activated). Pre-training processes "approximately 15T tokens across three stages": standalone ViT training, joint text-vision pre-training continuing "from a near-end Kimi K2 checkpoint over additional 15T vision-text tokens at 4K sequence length", then "mid-training on high-quality data and long-context activation", extending context by YaRN interpolation (§4.3).
- Vision pre-training data includes an "agent data" category: "GUI screenshots and action trajectories across desktop, mobile, and web environments, including human-annotated demonstrations" (App. B.3). The report does not state a token share for it.
- SFT: "synthesizing high-quality candidate responses from K2, K2 Thinking and a suite of proprietary in-house expert models", with human annotation and multi-stage verification (§4.4.1). Dataset size is not reported.
- Post-training RL organizes "RL domains not by input modality but by abilities—knowledge, reasoning, coding, agentic, etc." (§2.3). There is no separate specialist-distillation stage in the report.

## Zero-vision SFT (§2.2)
- Text-only SFT data alone activates visual tool use; "all image manipulations are proxied through programmatic operations in IPython".
- "Compared to zero-vision SFT, our preliminary experiments show that text-vision SFT yields much worse performance on visual, agentic tasks" (no numbers printed).

## RL objective (§4.4.2, Eq. 1)
- For each problem x, K responses are sampled from π_old. The loss is a squared term over `Clip(π_θ/π_old, α, β)·(r(x,y_j) − r̄(x)) − τ log(π_θ/π_old)`, summed over tokens and normalized by N, the total generated tokens in the batch.
- The clip is a token-level gradient mask: "policy gradients are computed normally for tokens with log-ratios within the interval [α, β], while gradients for tokens falling outside this range are zeroed out", and it "relies strictly on the log-ratio to explicitly bound off-policy drift, regardless of the sign of the advantages".
- Optimizer: MuonClip. Values of α, β, τ, K are not printed.

## Toggle budget control (§4.4.2, Eq. 2)
- Failure mode named: "length-overfitting", where "models trained under rigid budget constraints often fail to generalize to higher compute scales".
- Toggle alternates every m iterations between Phase0 (reward multiplied by the indicator that mean accuracy < λ or |y_i| ≤ budget(x)) and Phase1 (plain reward up to the maximum token limit).
- budget(x) is the ρ-th percentile of token lengths among correct responses, estimated once at the start of training and then fixed.
- Evaluated on K2 Thinking: "Toggle decreases output tokens by 25∼30% with a negligible impact on performance" (Figure 5), and transfers: trained on mathematics and programming only, it still shortens GPQA and MMLU-Pro outputs.

## Cross-modal generality evidence (§2.3, Table 2)
```
Benchmark        Before Vision-RL   After Vision-RL   Improvement
MMLU-Pro               84.7              86.4            +1.7
GPQA-Diamond           84.3              86.4            +2.1
LongBench v2           56.7              58.9            +2.2
```

## Agent Swarm and PARL (§3, §5.2)
- Decoupled architecture: "a trainable orchestrator and frozen subagents instantiated from fixed intermediate policy checkpoints"; subagent trajectories are excluded from the optimization objective. Stated reason: avoiding "credit assignment ambiguity and training instability".
- Reward: `r_PARL(x,y) = λ1·r_parallel + λ2·r_finish + r_perf(x,y)`. r_parallel counters "serial collapse" (defaulting to single-agent execution); r_finish counters "spurious parallelism", defined as spawning "many subagents without meaningful task decomposition". "λ1 and λ2 are annealed to zero over the course of training."
- CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t)); training and evaluation are constrained by critical steps rather than total steps.
- Table 6: BrowseComp 78.4 (Agent Swarm) vs 60.6 (single-agent K2.5); WideSearch Item-F1 79.0 vs 72.7; In-house Swarm Bench 58.3 vs 41.6. WideSearch execution time to a target Item-F1 falls by "3× ∼ 4.5×".
- Framed as "proactive context control", contrasted with reactive "Hide-Tool-Result", "Summary", or "Discard-all".

## Agentic RL environment (App. D)
- Gym-like interface with pluggable Toolset, Judge, and prompt-diversification modules composed with core agent loops.
- "A dedicated Rollout Manager orchestrates up to 100,000 concurrent agent tasks during the RL process, providing fine-grained control to enable features like partial rollout."
- "Our framework strictly follows a Token-in-Token-out paradigm. We also record log probabilities for all inference engine outputs to perform train-inference mismatch correction."
- An LLM Gateway proxy records rollout requests for black-box environments that only speak the standard LLM API protocol.

## Evaluation settings (App. E)
- Default: temperature 1.0, top-p 0.95, context length 256k. Reasoning benchmarks use a 96k completion budget; AIME 2025 and HMMT 2025 are Avg@64, GPQA-Diamond Avg@8.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2602.02276 (v2, 2026-08-07): Abstract, §1, §2.2, §2.3, §3, §4.1, §4.3, §4.4, §5.2, App. B.3, App. D, App. E.
- Not reported by the source: SFT dataset size; RL hyperparameters α, β, τ, K, learning rate, steps; per-stage token shares for agent data; number of RL environments or tasks; GPU count or hours.
