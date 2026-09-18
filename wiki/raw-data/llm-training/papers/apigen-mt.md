<!-- scope: agentic multi-turn data synthesis — task blueprints validated by execution, policy unit tests, and an LLM committee, then simulated human-agent trajectories on τ-bench; xLAM-2-fc-r models
     deps: [[apigen]]
     see-also: [[xlam]], [[bfcl]], [[toolace]], [[persona-hub]], [[api-bank]]
-->

# APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay
- **Core Insight:** Generating and validating task blueprints with an agentic feedback loop raised task-configuration success from 28% to 70%, and xLAM-2-70b-fc-r trained on the resulting multi-turn trajectories reached 56.2% average pass@1 on τ-bench retail and airline versus 38.2% for Llama 3.1 70B Instruct (Fig. 4, Table 2).
- **Guideline:** When building multi-turn tool-use SFT data in an executable environment with written policies, generate and validate the ground-truth actions first (execution check, policy unit tests, LLM committee) and then simulate the conversation, keeping only trajectories whose final state and outputs match, because this pipeline produced the τ-bench and BFCL v3 gains in Tables 1–2; the paper instantiates it only on τ-bench retail and airline.
- **Authors:** Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, et al. (Salesforce AI Research)
- **Year:** 2025 (arXiv v1 2025-04; v4 2025-07; preprint)
- **URL:** https://arxiv.org/abs/2504.03601 ; data: https://huggingface.co/Salesforce/APIGen-MT-5k ; models: https://huggingface.co/Salesforce/xLAM-2
- **Source type:** paper
- **Relevant topics:** agentic data synthesis, multi-turn tool use, simulated users, rejection sampling, behavioral cloning, function calling

## Abstract
Training agents for multi-turn interaction needs data with realistic human-agent dynamics, which is scarce and expensive to collect. APIGen-MT is a two-phase framework. Phase 1 produces task blueprints with ground-truth actions, using a committee of LLM reviewers and iterative feedback loops. Phase 2 turns each blueprint into a full trajectory through simulated human-agent interplay. The authors train the xLAM-2-fc-r models (1B to 70B parameters) and report that they outperform frontier models such as GPT-4o and Claude 3.5 on τ-bench and BFCL, with smaller models surpassing larger counterparts in multi-turn settings and higher consistency across trials. They release 5K synthetic trajectories and the models.

## Key Contributions
- A blueprint-then-dialogue pipeline that uses environment execution feedback and a review committee (§1, §3.2).
- Reverse task recombination: build long tasks by concatenating validated shorter tasks (§4.1.3).
- A stabilized simulated user: Best-of-N sampling (N=4) with self-critique (§4.2).
- Release of APIGen-MT-5k and xLAM-2-fc-r models trained on Llama 3.1/3.2 and Qwen 2.5 (§1).

## Key Figures/Tables to Study
- Fig. 2 (framework) and Fig. 3 (τ-bench realization with diff patches and state/output rewards).
- Fig. 4 (success rates and trajectory statistics) and Fig. 5 (turn distribution).
- Table 1 (BFCL v3), Table 2 (τ-bench), Fig. 6 (pass^k), Table 3 (naive vs BoN user), Fig. 7 (success by task length).
- App. B Figs. 8–12: generation, alignment-judge, review, trajectory, and BoN-user prompts.

## Technical Details
- **Formulation:** a POMDP (U, S, A, O, T, R) with actions {tool_call, response}; reward depends on the cumulative environment state change and the assistant's responses (§3.1).
- **Phase 1 context (§4.1.1):** each τ-bench domain's APIs form a directed graph (edge A→B if B's inputs can depend on A's output and the pair is allowed by policy); tasks come from random walks plus five samplers: API (read vs write; ground truth centers on write APIs), policy, domain data, persona (PersonaHub), and few-shot examples. The generator outputs thought, instruction q, actions a_gt, outputs o_gt.
- **Validation (§4.1.2):**
  1. Action validation: format check; execution check in the environment, recording the state change as a diff_patch; policy compliance via domain policies translated into Python unit tests.
  2. Alignment validation: a committee of LLM judges scores Correctness, Completeness, Satisfaction, Creativity as 0/1 each (App. B Fig. 9), aggregated by majority vote.
  3. Final review: tasks with average score above a predefined threshold are accepted; failures get a summarized improvement plan and are regenerated (reflection).
- **Feedback budget:** at most 3 reflection turns for retail and 5 for airline (§4.3).
- **Reverse task recombination:** combine tasks with the same persona, concatenate actions and outputs, rerun the policy check, write a combined instruction, and revalidate from Stage 2 (§4.1.3).
- **Phase 2 (§4.2):** a persona-guided human LM reveals the task in turns; the agent is gpt-4o in function-calling mode; a trajectory is kept only if the final environment state matches a_gt and the responses match o_gt (r = 1). Each task is attempted up to 3 times; all unique successful trajectories are kept.
- **Collection (§4.3):** 15 read and 13 write APIs across retail and airline; gpt-4o and DeepSeek V3 are used for generation, validation, and interplay.
- **Statistics (Fig. 4, §4.3):** Phase 1 success 70% with agentic feedback vs 28% without; Phase 2 trajectory success 67%; turns per trajectory min 1, max 29; average 7 tool calls and 6 user turns per trajectory. The text states that gpt-4o takes an average of 12 turns to complete a task.
- **BFCL v3 (Table 1, leaderboard of 04/03/2025):** overall accuracy xLAM-2-70b-fc-r 78.19 (rank 1), 32b 75.83 (rank 2), 8b 72.83 (rank 4), 3b 65.11 (rank 14), 1b 58.90 (rank 36); multi-turn 75.12 / 66.38 / 69.25 / 56.00 / 43.12, vs GPT-4o-2024-11-20 (FC) 41 and o1 (Prompt) 36.
- **τ-bench (Table 2, pass@1, ≥5 trials, naive user, think tool only), retail / airline / overall:** xLAM-2-70b 67.1 / 45.2 / 56.2; 32b 64.3 / 45.0 / 54.6; 8b 58.2 / 35.2 / 46.7; 3b 44.4 / 32.0 / 38.2; 1b 22.5 / 21.0 / 21.8; Llama 3.1 70B Instruct 50.4 / 26.0 / 38.2; gpt-4o-2024-11-20 62.8 / 43.0 / 52.9; Claude 3.5 Sonnet 62.6 / 36.0 / 49.3; Claude 3.5 Sonnet (new) 71.5 / 48.8 / 60.1; o1 73.5 / 54.2 / 63.9.
- **Consistency:** on airline, xLAM-2-70b-fc-r has a higher pass^5 than Claude 3.5 Sonnet (new) despite a lower pass^1 (§5.3, Fig. 6). The BoN user raises retail success and lowers variance over 5 trials: gpt-4o 62.8 → 67.0 (variance 11.1 → 2.6), xLAM-2-70b 67.1 → 68.8 (9.7 → 4.0) (Table 3).
- **Released models (model card, not paper):** the Llama-xLAM-2-70b-fc-r README lists 128k context for Llama-xLAM-2-70b-fc-r and Llama-xLAM-2-8b-fc-r, and 32k default (max 128k with YaRN) for xLAM-2-32b/3b/1b-fc-r, with a footnote that these defaults apply to Qwen-2.5-based models; license cc-by-nc-4.0.
- **Task length:** tasks split at the 33rd/66th percentiles of Claude 3.5 turns (short <13, medium 13–18, long >18); on long tasks xLAM-2-70b succeeds more often than gpt-4o and less often than Claude, and needs more user interactions than Claude (§5.4, Fig. 7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| xLAM-2-fc-r series | 1B, 3B, 8B, 32B, 70B | SFT | Base models | Llama 3.1/3.2 Instruct and Qwen 2.5 Instruct; per-size mapping not stated in the paper | arXiv:2504.03601v4 §1, §5.1 | verified 2026-09-14 | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | Objective and masking | filtered behavioral cloning; trajectories split at every assistant response; loss on assistant tokens only | §5.1 | verified 2026-09-14 | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | Data mixture | APIGen-MT trajectories + APIGen function-calling data [26] + agentic data from [52, 53]; proportions not reported | §5.1 | verified 2026-09-14 (sources) / not reported (proportions) | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | Epochs | at most 3 | §5.1 | verified 2026-09-14 | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | Method, precision, optimizer | full fine-tuning with LLaMA-Factory, DeepSpeed ZeRO stage 3, FlashAttention 2, bfloat16, AdamW | §5.1 | verified 2026-09-14 | no ablation reported |
| xLAM-2-fc-r series | 1B–70B | SFT | Compute | one NVIDIA H200 node | §5.1 | verified 2026-09-14 | — |
| xLAM-2-fc-r series | 1B–70B | SFT | LR, schedule, batch, sequence length | not reported | checked body, App. A–B, Llama-xLAM-2-70b-fc-r model card | not reported | — |

## Findings relevant to agentic training, negative feedback, generality, distillation
- **Negatives are discarded:** Phase 2 keeps only successful trajectories (rejection sampling); the authors state that failed trajectories may offer "additional contrastive signal" in future work (§4.2, §6). Phase 1 failures are summarized into feedback for regenerating the task (§4.1.2), and only validated tasks proceed to Phase 2 (§3.2.2).
- **Generality is not measured beyond tool use:** evaluation covers BFCL v3 and τ-bench only; no general-capability benchmark is reported (§5.1–5.4). The τ-bench evaluation uses the same two domains, APIs, and policies used for data generation (§4.3, §5.1); overlap checks against τ-bench test tasks are not reported. Other agentic data is mixed in "to enhance the dataset diversity" (§5.1).
- **Distillation:** trajectories are produced with gpt-4o and DeepSeek V3 and used to train smaller models; the authors describe the result as knowledge transfer that lets xLAM-2-8b-fc-r (46.7) exceed Llama 3.1 70B Instruct (38.2) on τ-bench (§5.2, Interpretation).
- **Long context:** trajectories reach 29 turns (Fig. 4); token lengths per trajectory are not reported.
- **Limitations stated by the authors:** simulated-user stochasticity remains, multi-stage validation adds computational overhead (no number given), and extension to more domains and RL-based self-improvement are future work (§6).

## Connections
- [[apigen]] — APIGen-MT extends the single-turn APIGen pipeline with agentic feedback and simulated interplay (§3.2); its data is also in the training mix (§5.1).
- [[xlam]] — agentic data from xLAM [52] is mixed into training; xLAM-2-fc-r is the model family trained here.
- [[bfcl]] — BFCL v3 is one of the two evaluation benchmarks (Table 1).
- [[toolace]] — ToolACE-2-8B (68.39) and ToolACE-8B (58.42) appear as BFCL v3 baselines (Table 1).
- [[persona-hub]] — source of personas for the Persona Sampler (§4.1.1).
- [[api-bank]] — earlier tool-use data pipeline whose filtering relies on one LLM tester agent without environment execution.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2504.03601 (arXiv v4, 2025-07-19) and the Llama-xLAM-2-70b-fc-r Hugging Face model card.
- Corrections to the previous card version:
  - "collected tasks average around 12 turns" → the text states gpt-4o takes an average of 12 turns per task (§4.3); Fig. 4 reports an average of 6 user turns and 7 tool calls per trajectory (min 1, max 29 turns). Both are kept with loci.
  - "agentic feedback improves task-collection success rate to about 70%" → 70% with vs 28% without agentic feedback (Fig. 4).
  - "final semantic review uses committee aggregation" → committee scoring with majority vote is Stage 2; Stage 3 applies a score threshold and sends consolidated feedback (§4.1.2).
  - BFCL v3 for xLAM-2-8b-fc-r: the 69.25% multi-turn value is correct; overall accuracy is 72.83 at rank 4 (Table 1). The earlier audit lead of "72.08 (rank 4)" was not confirmed: 72.08 belongs to GPT-4o-2024-11-20 (Prompt) at rank 5.
  - "outperform GPT-4o and Claude 3.5 on τ-bench" → xLAM-2-70b (56.2) exceeds gpt-4o (52.9) and Claude 3.5 Sonnet (49.3) but not Claude 3.5 Sonnet (new) (60.1) (Table 2).
  - "Simulation gap", "Schema dependence", "Validation overhead" risk bullets → replaced by the limitations the authors state (§6).
- Removed as unsupported by the source: lineage links to [[self-instruct]], [[magpie]], [[evol-instruct]], [[oss-instruct]], [[rlvr-tulu3]] (not cited by the paper); "substantial filtering and review cost" (no cost number reported).
- Not reported by the source: token lengths; SFT learning rate, batch size, sequence length; data mixture proportions; committee member models and the Stage 3 threshold value; per-domain trajectory counts; general-capability evaluation. No τ-bench card exists in this library for the user simulator and state-based evaluation used here.
