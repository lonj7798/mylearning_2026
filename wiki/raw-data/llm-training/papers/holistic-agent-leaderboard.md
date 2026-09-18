<!-- scope: HAL (arXiv:2510.11977) — standardized parallel agent-evaluation harness, 21,730 rollouts over 9 models x 9 benchmarks x scaffolds with cost-accuracy Pareto analysis, and LLM-aided (Docent) log analysis of shortcuts, unsafe actions, and scaffold bugs; evaluation only, no training
     deps: [[helm]]
     see-also: [[agentic-benchmark-checklist]], [[tau-bench]], [[tau2-bench]], [[adding-error-bars-evals]], [[impossiblebench]], [[swe-bench-illusion]]
-->

# Holistic Agent Leaderboard: The Missing Infrastructure for AI Agent Evaluation
- **Core Insight:** Across 21,730 rollouts (9 models, 9 benchmarks, about $40,000), higher reasoning effort gave equal or lower accuracy in 21 of 36 model-agent-benchmark combinations, and LLM-aided log analysis found shortcuts (for example agents finding benchmark answers on HuggingFace or arXiv) and a test-set leak in an official TAU-bench scaffold that accuracy alone did not reveal (Abstract, §4.1 item 5, §4.2, App. A5).
- **Guideline:** When comparing agents or checkpoints on agentic benchmarks, report cost or tokens next to accuracy, hold the scaffold fixed or evaluate several, and inspect trajectories for shortcuts, instruction violations, and environment failures before trusting a score, because in HAL two Online Mind2Web agents with different scaffolds differed 9x in cost at a two-point accuracy difference, and a leaked few-shot file invalidated one scaffold's results (§4.1 item 6, App. A5).
- **Authors:** Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S. Siegel, Boyi Wei, et al. (last authors Percy Liang, Arvind Narayanan; Princeton and collaborators)
- **Year:** 2025 (arXiv v1 2025-10; preprint)
- **URL:** https://arxiv.org/abs/2510.11977 ; harness: https://github.com/princeton-pli/hal-harness ; traces: https://huggingface.co/datasets/agent-evals/hal_traces ; leaderboard: hal.cs.princeton.edu
- **Source type:** paper
- **Relevant topics:** agent evaluation, scaffolds, cost-accuracy Pareto frontier, reasoning effort, log analysis, benchmark gaming, contamination

## Abstract
Agent evaluations are slow, non-standardized, rarely report cost, and rarely detect shortcuts or unsafe actions. HAL makes three contributions. First, a standardized harness orchestrates parallel evaluations across hundreds of VMs, reducing evaluation time from weeks to hours. Second, a three-dimensional analysis over models, scaffolds, and benchmarks: 21,730 rollouts across 9 models and 9 benchmarks in coding, web navigation, science, and customer service, costing about $40,000; one finding is that higher reasoning effort reduced accuracy in the majority of runs. Third, LLM-aided log inspection surfaces unreported behaviors such as searching for the benchmark on HuggingFace instead of solving the task, or misusing credit cards in flight booking. All logs (2.5B tokens of LM calls) are released (Abstract).

## Key Contributions
- Harness: scaffolds expose a minimal `run(input, **kwargs) -> dict` interface; Weave logging, LiteLLM for cross-provider calls, and local / Docker / Azure VM execution behind one interface (§2, Table 2, App. A2).
- Leaderboard with accuracy-vs-dollar-cost and accuracy-vs-token Pareto frontiers per benchmark (§4.1, Fig. 2, Fig. A2).
- Docent rubric analysis of transcripts with six categories: instruction violations, tool-use failures, self-correction, verification, environmental barriers, shortcuts/gaming (§4.2, Table A5).
- A list of 12 practical hurdles in large-scale agent evaluation (App. A3) and documented limitations (App. A4).

## Key Figures/Tables to Study
- Table 1 (prior work rarely ran these model-benchmark pairs), Table 3 (benchmarks and scaffolds), Table A11 (model prices).
- Fig. 2 (cost-accuracy Pareto frontiers), Fig. 3 (effect of higher reasoning), Fig. A3 (task-specific vs generalist scaffolds).
- Fig. 4 and Tables A3–A4 (failure-mode prevalence and reliability correlates), Table A2 (Docent precision), Tables A6–A10 (examples).

## Technical Details
- **Benchmarks (§1, Table 3):** web navigation (Online Mind2Web, AssistantBench, GAIA), coding (SWE-bench Verified Mini, USACO), science (CORE-Bench Hard, ScienceAgentBench, SciCode), customer service (TAU-bench Airline). SWE-bench Verified Mini has 50 tasks versus 500 in SWE-bench Verified; GAIA and AssistantBench use public test sets (App. A4.1).
- **Models (§3, Table A11):** o3, GPT-4.1, GPT-5 Medium, Claude-3.7 Sonnet, Claude Opus 4.1, DeepSeek V3, DeepSeek R1, o4-mini (Low/High), Gemini 2.0 Flash; Claude Sonnet 4 replaces Opus 4.1 on Online Mind2Web, whose estimated Opus cost was about $20,000 (Fig. 2 caption, App. A4.1). Prices range from $15/$75 (Opus 4.1) to $0.1/$0.4 (Gemini 2.0 Flash) per million input/output tokens, as of 2025-09-24 (§3).
- **Reasoning settings (§3 footnote, App. A3 item 8):** "high" for Anthropic models is LiteLLM's default of 4,096 reasoning tokens; OpenAI does not disclose its budgets.
- **Scaffolds (§3, App. A8):** task-specific scaffolds from prior work per benchmark, plus a HAL Generalist Agent (smolagents CodeAgent, planning interval 4 steps, at most 200 steps; tools: Google search, browsing, Python, bash, text inspector, file editor, VLM querier) run on CORE-Bench Hard, TAU-bench Airline, and SWE-bench Verified Mini.
- **Evaluation matrix (App. A4.1):** 142 model-scaffold-benchmark combinations reported in §4, from 186 runs. Most evaluations are single runs without confidence intervals because of cost (App. A3 item 1).
- **Cost-accuracy results (§4.1):** the most costly model run is on the Pareto frontier in 1 of 9 benchmarks (item 1); fewer than one-third of models are on the frontier on average; Gemini 2.0 Flash is on it in 7 of 9, GPT-5 and o4-mini Low in 4 of 9, DeepSeek R1 in 0 of 9 (item 2, Fig. 2). Token usage correlates positively with accuracy on 6 of 9 benchmarks (item 3). Opus 4.1 is on the token frontier in 3 of 8 benchmarks but the cost frontier once (item 4). On ScienceAgentBench o4-mini scores 27% at about 5x lower cost than GPT-5 at 30% (§3).
- **Reasoning effort (§4.1 item 5, Fig. 3):** pairs Claude Opus 4.1, Claude Sonnet 4, Claude-3.7 Sonnet (no vs high) and o4-mini (low vs high); 21 of 36 model-agent-benchmark combinations show equal or lower accuracy with more reasoning. The abstract words this as "reducing accuracy in the majority of runs".
- **Scaffolds (§4.1 items 6–7):** on Online Mind2Web, SeeAct with GPT-5 Medium costs $171 and Browser-Use with Claude Sonnet 4 costs $1,577, with a two-point accuracy difference; Claude models do better with Browser-Use, OpenAI models with SeeAct. Task-specific scaffolds beat the generalist on 9 of 12 runs (CORE-Bench Hard) and 11 of 12 (SWE-bench Verified Mini); the generalist costs less in 20 of 24 comparisons.
- **Benchmark cost (§4.1 item 8):** average per evaluation ranges from $13 (ScienceAgentBench) to over $450 (Online Mind2Web).
- **Docent setup (App. A7.1–A7.2):** GPT-5 Medium as judge; 48 model-scaffold pairs and 2,184 transcripts; after removing TAU-bench, 36 pairs and 1,634 transcripts. Human-validated precision: 0.87 (AssistantBench instruction following, n=49, inter-LLM κ 0.82 with Claude Sonnet 4), 1.00 (CORE-Bench verification, n=31), 0.94 (TAU-bench instruction following, n=36) (Table A2). Only precision is measured, not recall.
- **Log findings (§4.2, Tables A3–A4):** eight cases of agents finding gold answers on HuggingFace or arXiv (item 1, Table A8); hard-coded "plausible" solutions on CORE-Bench and SciCode (Tables A7, A10). On failed tasks, instruction violations occur in 67.0% (AssistantBench) and 62.8% (CORE-Bench); environmental barriers in 56.4% (AssistantBench), 43.8% (SciCode), and 40.3% (CORE-Bench). At least one tool-call failure occurs in 97.7% of failed and 100% of successful SciCode runs and 89.7% / 83.9% on CORE-Bench. Success is more likely with self-correction (RR 1.47, 1.54, 2.97 for AssistantBench, SciCode, CORE-Bench) and verification (RR 1.39, 1.87, 1.13); the §4.2 text summarizes these as 1.5x–4x and 13%–87%.
- **Unsafe actions (Table A6):** from the TAU-bench Few Shot agent logs, e.g. a purchase with an incorrect credit card that cannot be reversed (Claude Opus 4.1) and a $2,010 charge against a $1,000 budget to a wrong payment method (Claude Opus 4.1 high).
- **Leak (App. A5):** the official TAU-bench few-shot file `few_shot_data/MockAirlineDomainEnv-few_shot.jsonl` contained test-set examples; discovered after about $1,000 of evaluations; all results from that scaffold were excluded.
- **Infrastructure hurdles (App. A3):** a provider switched a DeepSeek R1 endpoint to R1-0528 under the same name; OpenRouter could serve FP4 on one call and FP8 on another; silent rate-limit failures score as wrong answers; the o3/o4-mini API removed the `stop` argument; Anthropic's default spend limit was $5,000 per month.

## Findings relevant to generality, negative feedback, agentic training
- **Generality of scaffolds:** a single generalist scaffold loses accuracy against task-specific scaffolds on 9 of 12 and 11 of 12 runs, at lower cost in 20 of 24 comparisons (§4.1 item 7). Model-scaffold interactions exist: on Online Mind2Web, Claude models score higher with Browser-Use and OpenAI models with SeeAct (§4.1 item 6, Fig. A3c).
- **Measurement errors:** single runs without confidence intervals (App. A3 item 1), weight or quantization changes behind stable endpoints (App. A3 items 2, 4), task instructions entangled with scaffold instructions ("don't guess" lowered Claude Opus 4.1 accuracy on AssistantBench; §1, App. A3 item 10), leaked few-shot data (App. A5), and SWE-Agent cost accounting that ignores cache hits (App. A4.1).
- **Negative signals:** failures are analyzed, not trained on; the paper states it cannot establish whether fixing a flagged failure would lead to success, which would require checkpointing and replay (App. A4.2). The paper does not train models.

## Connections
- [[helm]] — HAL cites HELM and LM Evaluation Harness as the LM-level standardization it extends to agents (§1).
- [[agentic-benchmark-checklist]] — cited as Zhu et al. 2025a for construct validity when selecting benchmarks (§3).
- [[tau-bench]] / [[tau2-bench]] — HAL uses the original TAU-bench Airline; App. A4.1 notes τ²-bench fixes known task and metric issues.
- [[adding-error-bars-evals]] — HAL reports single runs without confidence intervals because of cost (App. A3).
- [[impossiblebench]], [[swe-bench-illusion]] — other sources in this library on agents exploiting tests or benchmark memorization.
- [[openhands-data]] — OpenHands appears in HAL's App. A9 survey as a scaffold used in prior GAIA and SWE-bench evaluations; HAL does not evaluate it.
- [[minimax-m2-aligning-to-what]], [[qwen3-coder-next]] — library sources on training agent models (MiniMax M2 agent generalization; Qwen3-Coder-Next), to compare against HAL's scaffold-dependence results; HAL does not cite them.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.11977 (arXiv v1, 2025-10-13; only version listed).
- Audit claims not found in the source: "LLM-aided log inspection of 2.5B tokens" → 2.5B tokens is the size of the released logs; the Docent analysis covered 2,184 transcripts on four benchmarks, 1,634 after removing TAU-bench (§1, App. A7.1). "Near-universal tool-call failures" → stated for SciCode and CORE-Bench only, two of the three analyzed benchmarks (§4.2 item 3). The link to Qwen3-Coder-Next and MiniMax cross-scaffold findings is a course cross-reference, not in the paper. The credit-card example comes from the TAU-bench Few Shot agent logs, the scaffold later excluded for leakage (Table A6, App. A5).
- Not reported by the source: latency results (App. A4.2), recall of Docent rubrics, multiple-seed variance for most runs.
