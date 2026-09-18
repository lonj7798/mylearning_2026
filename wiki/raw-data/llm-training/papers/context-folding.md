<!-- scope: Context-Folding (arXiv:2510.11967, ByteDance Seed / CMU / Stanford) — branch/return tools that let an agent fold finished sub-trajectories out of its working context; FoldGRPO (GRPO over folded contexts plus token-level process penalties) on Seed-OSS-36B-Instruct for BrowseComp-Plus and SWE-Bench Verified
     deps: [[grpo]], [[browsecomp-plus]]
     see-also: [[resum]], [[supo-summarization-rl]], [[memagent]], [[mem1]], [[dapo]], [[seed-oss-36b]], [[anthropic-multi-agent-research-system]]
-->

# Scaling Long-Horizon LLM Agent via Context-Folding
- **Core Insight:** A Seed-OSS-36B-Instruct folding agent with a 32K active context and up to 10 branches, trained with FoldGRPO, reaches pass@1 0.620 on BrowseComp-Plus and 0.580 on SWE-Bench Verified, above a 327K-context ReAct agent trained with GRPO (0.540 / 0.574) and a 32K × 10 summary agent trained with GRPO (0.527 / 0.550) (Table 1).
- **Guideline:** When an agent must manage a bounded working context through branch/return tools under a binary outcome reward, add token-level penalties for token-heavy main-thread work, out-of-scope branches, and failed tool calls, because on the same folding agent plain GRPO reached 0.567 / 0.564 with finish rates 0.738 / 0.612, while FoldGRPO reached 0.620 / 0.580 with finish rates 0.935 / 0.962 (Tables 1-2); this was tested with one 36B base model on two task families.
- **Authors:** Weiwei Sun, Miao Lu, Zhan Ling, Kang Liu, Xuesong Yao, Yiming Yang, Jiecao Chen (ByteDance Seed; Carnegie Mellon University; Stanford University)
- **Year:** 2025 (arXiv v1 2025-10; preprint)
- **URL:** https://arxiv.org/abs/2510.11967
- **Source type:** paper
- **Relevant topics:** agentic RL, long-horizon agents, context management, GRPO variants, process rewards, deep research, SWE agents, KV-cache, asynchronous rollout

## Abstract
LLM agents on long-horizon tasks are limited by context length. Context-Folding lets an agent branch into a sub-trajectory for a subtask and fold it on completion: the intermediate steps are removed and a concise summary of the outcome is kept. FoldGRPO is an end-to-end RL framework with process rewards that train task decomposition and context management. On Deep Research and SWE tasks the folding agent matches or outperforms ReAct baselines while using an active context 10× smaller, and it outperforms summarization-based context management (Abstract).

## Key Contributions
- Two context tools: `branch(description, prompt)` opens a separate working context for a sub-task; `return(message)` folds the branch and appends only the templated message to the main thread (§2.2, App. C.3).
- FoldGRPO: GRPO computed on folded histories, with tool-observation tokens masked and a token-level process reward added to the outcome reward inside the advantage (§2.3.1).
- Three process penalties aimed at two failure modes observed with outcome reward only: token-intensive work left in the main context, and branches that are not returned after the sub-task ends (§2.3.2).
- Comparisons against ReAct agents at 32K and 327K context and a summary agent, all trained with the same base model, data, infrastructure and hyperparameters (§3.3, Table 1).

## Key Figures/Tables to Study
- Fig. 2 (folding and FoldGRPO), Eq. 1 (folded-context policy), the §2.3.1 objective and advantage.
- Table 1 (main results, with 100B+ ReAct references), Table 2 (behavior statistics: Finish, Main Len, Scope, #Branch).
- Fig. 3 (by difficulty), Fig. 4 (training dynamics), Fig. 5 (context length and combined questions), Fig. 7 (107K → 6K case), Fig. 8 (training time).

## Technical Details
- **Folded policy (Eq. 1):** p(τ | q) = Π_i π_θ(a_i | q, F(τ_<i)). a_i is the LLM output at step i (reasoning and tool call), o_i the tool result, τ_<i all earlier action-observation pairs, F the context manager that removes the pairs between each branch and its return (§2.1-2.2).
- **Inference:** on `return`, the KV-cache is rolled back to the branch position, where the prefix equals the context before the branch call (§2.2). Plan-execution instantiation: branching is disabled inside a branch, so there are no nested branches (§2.2).
- **Objective (§2.3.1):** J = E[(1/Σ_i|τ_i|) Σ_i Σ_t min(r_{i,t}(θ) Â_{i,t}, clip(r_{i,t}(θ), 1−ε_low, 1+ε_high) Â_{i,t})]. r_{i,t} = π_θ(τ_{i,t} | q, F(τ_{i,<t})) / π_θold(τ_{i,t} | q, F(τ_{i,<t})) · 1^LLM_{τ_{i,t}}; the indicator keeps only LLM-generated tokens. G trajectories per task; R_i ∈ {0, 1}.
- **Advantage (§2.3.1):** Â_{i,t} = (clip(R_i + Q_{i,t}, 0, 1) − mean({R_i})) / std({R_i}). Q_{i,t} is the token-level process reward. Derived from this formula: for a failed trajectory (R_i = 0) a negative Q leaves clip(·) at 0, so the penalties change advantages only inside successful trajectories; a token with Q = −1 in a successful trajectory receives the same advantage as a failed trajectory's tokens.
- **Process rewards (§2.3.2):** unfolded-token penalty Q = −1 on all main-thread tokens, except tokens in branch-creating turns, when the main thread exceeds 50% of the working context limit; out-of-scope penalty Q = −0.2 on all tokens of a branch that GPT-5-nano judges, from the branch prompt and returned message, to have acted outside its sub-task; failure penalty Q = −1 on all tokens of a failed tool-call turn; Q = 0 otherwise.
- **Training implementation:** each branch is kept as a separate causally conditioned sequence instead of one concatenated sequence, which the authors state is not directly compatible with existing infrastructure such as verl (§3.2, App. A.1).
- **Datasets (§3.1):** BrowseComp-Plus split into 680 training and 150 evaluation instances; tools `search(query, topk)` and `open_page(url)`; reward from the official LLM-based judge; retriever Qwen3-Embed-8B, top-k default 10, first 512 tokens shown per result, opened pages truncated to 4,096 tokens (App. C.1). SWE: 740 training instances from SWE-Gym and SWE-Rebench subsets where 8 rollouts of the baseline agent (Seed-OSS-36B-Instruct with OpenHands, response length 65,536) succeed between 0 and 87.5% of the time; evaluation on SWE-Bench Verified (N = 500); tools `execute_bash`, `str_replace_editor`, `think`; reward from unit tests in a sandbox (§3.1, footnote 1, App. C.2).
- **Difficulty bins:** BrowseComp-Plus by ReAct acc@8 (easy ≥ 87.5%, hard 0%), 50 instances each; SWE-Bench Verified by time-to-resolve: easy 194, medium 261, hard 45 (§3.1).
- **Main results, pass@1 BC-Plus / SWE-V (Table 1):** Seed-OSS-36B ReAct 32K 0.286 / 0.436, + GRPO 0.446 / 0.480; ReAct 327K 0.478 / 0.552, + GRPO 0.540 / 0.574; Summary 32K × 10 0.386 / 0.488, + GRPO 0.527 / 0.550; Folding 32K × 10 0.420 / 0.492, + GRPO 0.567 / 0.564, + FoldGRPO 0.620 / 0.580. Tool calls for the FoldGRPO agent: 19.2 / 96.5. References: GPT-5 ReAct 327K 0.793 / 0.718; DeepSeek-V3.1 0.613 / 0.610.
- **RL gain:** +20.0 (BC-Plus) and +8.8 (SWE) pass@1 points, from the untrained folding agent to FoldGRPO (§1, §4.1; 0.420 → 0.620 and 0.492 → 0.580 in Table 1). Relative to the untrained 327K ReAct agent the FoldGRPO agent is +14.2 and +2.8 (Table 1 parentheses).
- **Behavior (Table 2), Finish / Main Len / Scope / #Branch:** BC-Plus untrained 0.806 / 12,195 / 0.774 / 3.51; GRPO 0.738 / 22,285 / 0.762 / 3.88; FoldGRPO 0.935 / 7,752 / 0.895 / 4.98. SWE untrained 0.781 / 47,475 / 0.473 / 3.05; GRPO 0.612 / 48,908 / 0.419 / 3.80; FoldGRPO 0.962 / 8,885 / 0.754 / 5.90. The main trajectory is about 8K tokens while over 100K tokens are processed (§4.3).
- **Dynamics (§4.2, Fig. 3-4):** RL gains are larger on medium and hard instances; tool calls, branches, response tokens and searched pages increase during training; on the hard BC-Plus subset response length rises from about 100K to over 160K tokens.
- **Scaling (§4.4, Fig. 5):** varying branches from 0 to 16, the method stays above ReAct and plateaus beyond 320K tokens. With 1 to 50 combined questions, unlimited branching and a 1M-token ReAct limit, the agent trained with at most 10 branches uses an average of 32.6 branches on 50-question tasks.
- **Other analyses:** a case study folds a 107K-token trajectory into 6K with 4 branches (§4.5.1, Fig. 7); the 327K ReAct model is slower per rollout (1.52×) and per training step (1.43×) (Fig. 8); a parallel-branch variant scores 0.6133 on BC-Plus, creates about 2.3 parallel branches and reads 110 pages vs 80, without beating the single-branch agent (§4.5.3). The authors' system prompt gives 0.478 on BC-Plus versus about 0.08 for the benchmark's default prompt with Seed-OSS-36B (App. B.1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Folding Agent on Seed-OSS-36B-Instruct + FoldGRPO | 36B | RL | Framework; rollout batch; group size; PPO batch | VeRL; 32; 8; 128 | arXiv:2510.11967v1 §3.2 | verified 2026-09-14 | no ablation reported |
| same | 36B | RL | Learning rate; KL; clip | 1 × 10⁻⁶; no KL term; clip low 0.2, clip high 0.28 | §3.2 | verified 2026-09-14 | no ablation reported |
| same | 36B | RL | Training length | 50 steps (about 2 epochs) | §3.2 | verified 2026-09-14 | no ablation reported |
| same | 36B | RL | Rollout | asynchronous; main rollout stops at 95% of prompts, rest finished by a standalone process; max off-policy steps 5 | §3.2, App. A.2 | verified 2026-09-14 | App. A.2: "no performance degradation" vs fully on-policy (no numbers) |
| same | 36B | RL | Context budget | 32,768 max LLM context; up to 10 branches; theoretical max 327,680 tokens | §3.2 | verified 2026-09-14 | Fig. 5 left: 0-16 branches, plateau beyond 320K |
| same | 36B | RL | Reward; loss mask | binary outcome R ∈ {0,1} (LLM judge / unit tests) plus Q ∈ {−1, −0.2, 0}; tool-observation tokens masked | §2.3.1-2.3.2, §3.1 | verified 2026-09-14 | Tables 1-2: FoldGRPO vs GRPO on the same agent |
| same | 36B | RL | Unfolded-token threshold; scope judge | main thread > 50% of working context limit; GPT-5-nano | §2.3.2 | verified 2026-09-14 | no ablation of threshold or penalty size reported |
| same | 36B | RL | Training prompts | BC-Plus 680 instances; SWE 740 instances (baseline success in 0-87.5% over 8 rollouts) | §3.1 | verified 2026-09-14 | no ablation reported |
| same | 36B | eval-gate | Decoding | greedy, temperature 0 | §3.2 | verified 2026-09-14 | — |
| same | 36B | RL | Optimizer, LR schedule, warmup, per-turn response limit, compute, joint vs per-task training | not reported (checked body, App. A-C) | — | not reported | — |

## Findings relevant to generality, negative feedback, long context, agentic training
- **Long context:** a 32K × 10 folding agent outperforms a 327K ReAct agent after RL on both tasks (Table 1) and uses more branches than seen in training on 50-question compound tasks (§4.4.2). The authors call this length generalization; it is measured only on BC-Plus-derived compound questions.
- **Negative signals:** failures enter as gradient in two ways: zero-reward trajectories receive below-mean group advantages, and process penalties lower advantages of penalized tokens in successful trajectories (§2.3.1-2.3.2; the second point is derived from the advantage formula). A split of the gain between positive and negative signals is not reported.
- **Agentic training:** with outcome reward only, the authors observed two failure modes (unfolded token-heavy work, branches not returned) (§2.3.2); GRPO lowered Finish and Scope relative to the untrained agent (Table 2).
- **Generality:** evaluation is on a held-out BC-Plus split and SWE-Bench Verified. No general-capability or out-of-domain evaluations are reported. The authors suggest breadth-first tasks such as WideSearch for parallel branching (§4.5.3).

## Connections
- [[grpo]] — FoldGRPO builds on GRPO's group-relative advantage (§2.3.1, ref [25]).
- [[dapo]] — compare its decoupled clip bounds; this paper sets clip low 0.2 and clip high 0.28 (§3.2) without citing DAPO.
- [[browsecomp-plus]], [[swe-gym]], [[swe-rebench]] — benchmark corpus and SWE training-instance sources (§3.1).
- [[resum]], [[memagent]] — cited as summary-based context management; the Summary Agent baseline follows them (§3.3, refs [34], [38]).
- [[supo-summarization-rl]] — same author group, RL with summarization-based context management (ref [19]).
- [[mem1]] — source of the combined-questions protocol (§4.4.2, ref [43]).
- [[anthropic-multi-agent-research-system]], [[cognition-dont-build-multi-agents]] — multi-agent designs; the paper frames folding as sub-agents created on the fly with a shared prefix (§2.4).
- [[seed-oss-36b]] — base model (§3.2); [[verl-grpo]] — training framework (§3.2).
- [[chroma-context-rot]], [[lost-in-the-middle]] — long-context degradation motivating the method (§1).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.11967 (arXiv v1, 2025-10-13; only version listed).
- Audit claims not found in the source: "+20.0 over the 327K ReAct baseline" and "+8.8" → these are gains of FoldGRPO over the untrained folding agent; over the untrained 327K ReAct agent the gains are +14.2 and +2.8 (Table 1). "FoldGRPO beats plain GRPO by 7.7 points (BC+)" → §4.1 text says +7.7%, but Table 1 gives 0.620 vs 0.567 (5.3 points); SWE 1.6 matches Table 1 (internal inconsistency in the paper). "A counterpart to DeepSeek-V3.2's finding that discarding tool history works" → DeepSeek-V3.2 is not mentioned in the paper.
- Not reported by the source: optimizer and LR schedule, per-turn response length limit, GPU count and total compute, whether one model was trained on both task families, the unit of Fig. 8's time axis.
