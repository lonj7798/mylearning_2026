<!-- scope: rStar2-Agent (Microsoft Research, 2025) — agentic RL with a Python execution tool; GRPO-RoC (Resample-on-Correct) rollout filtering; non-reasoning SFT then 3-stage RL on Qwen3-14B-Base
     deps: [[grpo]], [[deepseek-r1]]
     see-also: [[rstar-math]], [[rstar]], [[demystifying-agentic-rl]], [[reasoning-trap-tool-hallucination]], [[entropy-mechanism-llm-rl]], [[rlvr-beyond-base-model]], [[toolace]], [[apigen-mt]]
-->

# rStar2-Agent: Agentic Reasoning Technical Report
- **Core Insight:** Under a binary answer-only reward, correct trajectories that contain failed tool calls are still reinforced; oversampling 2G = 32 rollouts and keeping G = 16 — positives chosen with probability inversely proportional to a tool-error-plus-format penalty, negatives sampled uniformly — reduces tool errors in positive trajectories and lets a 14B model reach 80.6% AIME24 / 69.8% AIME25 pass@1 in 510 RL steps (§2.2.3, Fig. 4; §4.3; Table 3).
- **Guideline:** When RL uses an external tool whose errors cannot be penalized without reward hacking, keep the reward answer-only and move quality control into rollout selection: filter positives for clean tool use and format, but keep negatives unfiltered so failure diversity is preserved (§2.2.3, §4.3.1).
- **Authors:** Ning Shang, Yifei Liu, Yi Zhu, Li Lyna Zhang, Weijiang Xu, Xinyu Guan, et al. (Microsoft Research; 15 authors, project leads Li Lyna Zhang and Mao Yang)
- **Year:** 2025 (arXiv v1 2025-08-28; only version)
- **URL:** https://arxiv.org/abs/2508.20722 (code: https://github.com/microsoft/rStar)
- **Source type:** paper (technical report)
- **Relevant topics:** agentic RL, tool-integrated reasoning, GRPO variants, rollout filtering, non-reasoning cold start, multi-stage length schedule, RL infrastructure

## Abstract
rStar2-Agent trains Qwen3-14B-Base to reason with a Python execution tool through multi-turn RL. The report names three components: (i) an isolated code-execution service and a load-balanced rollout scheduler that make agentic RL feasible on 64 AMD MI300X GPUs; (ii) GRPO-RoC, which adds a Resample-on-Correct rollout strategy to GRPO to counter noise from tool errors under outcome-only reward; (iii) a recipe of non-reasoning SFT followed by three RL stages. The model reaches 80.6% on AIME24 and 69.8% on AIME25 with shorter responses than DeepSeek-R1, and is reported to transfer to GPQA-Diamond, BFCL v3, IFEval and Arena-Hard although RL data is math-only.

## Key Contributions
- GRPO-RoC: asymmetric down-sampling from 2G rollouts — negatives uniform, positives weighted by a tool-error and format penalty (§2.2.3, Eqs. 4–7).
- Measurement that under plain GRPO with tools the tool-error ratio inside positively rewarded trajectories plateaus near 15% (Qwen2.5-32B) and 10% (Qwen3-14B) (§2.2.2, Fig. 4).
- Non-reasoning SFT cold start (function calling, instruction following, chat only) before RL (§4.1, Table 1).
- Three-stage RL at 8K → 12K → 12K max length, with offline difficulty filtering only in stage 3 (§4.3, Table 2).
- Reported negative results: overlong filtering and n-gram repetition down-weighting both hurt (§4.3.1); RL collapse after the stage-3 peak (§5.3).

## Key Figures/Tables to Study
- Fig. 4: tool-error ratio in positive trajectories, GRPO vs GRPO-RoC. Fig. 9: RoC ablation (accuracy and response length).
- Table 1: capability profile after non-reasoning SFT. Table 2: recipe comparison with DAPO, ReTool, MiniMax, MiMo, Magistral.
- Table 5: out-of-domain results; numbers in parentheses are after SFT, before RL. Table 6: comparison on Qwen2.5-32B.

## Technical Details
- **Rollout format.** Multi-turn: assistant turn ends at EOS; a `<tool_call>` JSON block is executed and the result is appended under the user role as `<tool_response>`; repeats until an `<answer>` or T turns. Environment returns one of four outcomes: stdout, IPython-style output, error with traceback, or timeout (§2.1, Fig. 3).
- **Reward.** r ∈ {0, 1} from a rule-based math verifier on the `\boxed{}` answer; no tool-error or step reward (§2.2.1, Eq. 3).
- **GRPO changes.** KL penalty removed; Clip-Higher with ε_low = 0.2, ε_high = 0.28; entropy loss removed (§2.2.1, Eq. 6).
- **RoC selection.** From 2G rollouts: Ô_neg = ⌊|O_neg|/2⌋ sampled uniformly; half of O_pos sampled with probability inversely proportional to p_total = p_err + p_format, where p_err = error tool calls / all tool calls (0.5 if no tool call) and p_format = 1 if no `<answer>` tag, else min(1, (num answer tags − 1)/num turns) (§2.2.3, Eqs. 4–5).
- **Infrastructure.** Environment service on the cluster CPUs: master node task queue with 32 send workers batching up to 64 calls; each worker node runs 1024 execution workers. Up to 45K tool calls per step at 0.3 s mean latency. Answer verification is also offloaded to this service. The rollout scheduler assigns requests by available KV-cache capacity instead of static equal splits, because SGLang evicts half of in-progress rollouts on KV overflow (§3.1–3.2, Figs. 5–7). Built on VERL v0.2 and SGLang.
- **SFT data.** 165K function-call examples (117K from ToolACE-11K, APIGen-MT-5K, Glaive-function-calling-v2-101k, plus 48K Magicoder reformatted as JSON calls); 30K Tulu3 instruction-following examples with o4-mini rewritten responses; 27K Llama-Nemotron chat examples with o4-mini rewritten prompts (§4.1).
- **RL data.** >100K candidates (17K integer-answer DAPO, 93K AoPS via OpenMathReasoning, 937 Project Euler) → 42K after cleaning; integer answers only, because rule-based verifiers mis-judge equivalent algebraic forms; OpenMathReasoning answers kept only if Qwen3-32B matches the label at least twice in 16 samples (§4.2).
- **Stages.** Stage 1: 42K problems, 8K max length, steps 1–300; mean length 1K → ~4K; clip ratio briefly >10%. Stage 2: 12K max length; length 4K → 6K. Stage 3: offline filter with 8 rollouts per problem, remove all-correct → 17.3K problems; optimizer state reset, reference model updated; 125 steps; length 6K → 8K. Total 510 steps in one week (§4.3, Fig. 8). Stage 2 length is not stated directly; 510 − 300 − 125 = 85 steps by subtraction.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| rStar2-Agent-14B (Qwen3-14B-Base) | 14B | SFT | epochs; LR; warm-up; schedule; batch | 3; 5e-6; 4%; cosine; 128 | arXiv:2508.20722v1 §5.1 | verified 2026-11-24 | no ablation reported |
| rStar2-Agent-14B | 14B | RL | optimizer; LR; warm-up | AdamW; constant 1e-6; linear over 20 rollout steps | §5.1; §4.3 | verified 2026-11-24 | no ablation reported |
| rStar2-Agent-14B | 14B | RL | prompts per batch; rollouts sampled / kept | 512; 32 / 16 | §4.3 | verified 2026-11-24 | Fig. 9: RoC vs GRPO-with-tool (G = 16, no RoC, 300 steps) |
| rStar2-Agent-14B | 14B | RL | clip ε_low / ε_high; KL; entropy bonus | 0.2 / 0.28; none; none | §2.2.1, §2.2.3 | verified 2026-11-24 | no ablation reported |
| rStar2-Agent-14B | 14B | RL | rollout temperature; max turns T | 1.0; 10 (stages 1–2), 15 (stage 3) | §5.1 | verified 2026-11-24 | §5.3: raising T to 20 did not prevent post-peak collapse |
| rStar2-Agent-14B | 14B | RL | max response length per stage | 8K → 12K → 12K | §4.3, Table 2 | verified 2026-11-24 | Fig. 8: stage-2 extension improved AIME after stage-1 plateau |
| rStar2-Agent-14B | 14B | eval-gate | decontamination; eval sampling | 8-gram overlap removal; T=0.6, 30K tokens, 30 turns, 16 samples | §5.1 | verified 2026-11-24 | not applicable |

## Findings relevant to generality and negative feedback
- **Generality.** RL data is math-only. GPQA-Diamond rises 42.1 → 60.9 (after SFT → after RL); BFCL v3 63.1 → 60.8, IFEval 83.7 → 83.4, Arena-Hard 86.8 → 86.6, which the authors describe as maintained, not improved (§5.2, Table 5). Non-reasoning SFT lowers MATH-500 from 62.0 (base) to 57.4 while raising tool and chat scores (Table 1).
- **Negative samples.** RoC keeps negatives unfiltered to preserve "diverse failure modes" (§2.2.3). Truncated overlong rollouts keep reward 0; DAPO-style overlong filtering raised the overlong ratio, which the authors attribute to repetitive outputs losing their only negative signal (§4.3.1).
- **Reward design failures.** Step-level rewards and tool-error penalties were rejected as prone to reward hacking (§2.2.3, no experiment shown). N-gram repetition down-weighting removed legitimate repeated verification calls and lowered scores (§4.3.1).
- **Ceiling.** After the step-510 peak, continued RL collapsed policy and reward; temperature 1.2, longer length, T = 20, higher clip, and optimizer reset all failed. The authors hypothesize the base model's pretrained capacity is the limit (§5.3).
- **Entropy analysis.** In 64 sampled trajectories, top-20% entropy tokens are forking tokens and reflection tokens after tool responses; code tokens are low-entropy (§5.4, Figs. 10–11).

## Connections
- [[grpo]], [[deepseek-r1]]: base objective and outcome-only reward design.
- [[rstar-math]], [[rstar]]: earlier rStar work (MCTS-based); this report replaces search with agentic RL.
- [[rlvr-beyond-base-model]]: cited for the hypothesis that RL does not extend reasoning beyond pretraining (§5.3).
- [[toolace]], [[apigen-mt]], [[glaive-function-calling]]: SFT function-calling sources (§4.1).
- [[demystifying-agentic-rl]], [[reasoning-trap-tool-hallucination]]: other agentic-RL sources for comparison.

## Verification
- Checked on 2026-11-24 against https://arxiv.org/abs/2508.20722 (v1 PDF, full text) and the README of https://github.com/microsoft/rStar (repo exists; README names rStar2-Agent and points to RL training instructions).
- Not reported: total GPU-hours beyond "64 MI300X, one week"; stage-2 step count (derived above); KL/entropy curves; exact downsample weights when p_total = 0; number of RL rollouts filtered by the format penalty.
- Internal inconsistency in source: §3.1 says the service achieves "45 calls per step" next to "up to 45K tool calls"; the card uses 45K from Fig. 6 and §1. §4.3 says stage-3 filtering uses "the latest policy (from the final 385 steps of Stage 2)"; the meaning (step index vs count) is ambiguous and not carried as a number.
