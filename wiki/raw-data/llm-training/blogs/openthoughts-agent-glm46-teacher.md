<!-- scope: OpenThoughts-Agent launch post (Dec 2025) and its HF model/dataset cards — OpenThinker-Agent-v1 (Qwen3-8B): terminal-agent SFT on ~15.2K GLM-4.6 traces, task-source and teacher ablations on a 70-task dev set, small RL stage on ~720 filtered NL2Bash tasks
     deps: [[open-thoughts]]
     see-also: [[terminal-bench-2]], [[swe-smith]], [[skyrl-agent]], [[distillation-source-matters]], [[together-coderforge-agent-trajectories]]
-->

# Launching the OpenThoughts-Agent Project
- **Core Insight:** In the teacher ablation on the Freelancer task source, SFT traces from GLM-4.6 give a dev-set score of 17.2, while traces from GPT5-Nano, GPT5, and GPT5-Mini give 8.2, 8.0, and 7.6; the released Qwen3-8B student (OpenThinker-Agent-v1) scores 4.9 on Terminal-Bench 2.0, 15.7 on SWE-Bench Verified, and 17.3 on OpenThoughts-TB-Dev versus 0.0 / 0.7 / 5.7 for Qwen3-8B (blog teacher chart; model card table).
- **Guideline:** When choosing a teacher for agent-trajectory SFT, compare candidate teachers by the student's score on a held-out dev set rather than by the teacher's own benchmark score, because in this study the GPT5 teacher, described as the best model on TerminalBench, did not improve the student over GPT5-Nano, while GLM-4.6 gave "almost a 2x improvement" (blog "SFT Data - Teacher Model"; RL dataset card). The comparison uses one task source and one student size.
- **Authors:** OpenThoughts-Agent Team (citation block: "Team, OpenThoughts-Agent"); collaboration listed as Stanford, UC Berkeley, UT Austin, NYU, UW, UCLA, UNC, TUM, LAION, and others
- **Year:** 2025 (blog dated 2025-12-05; no arXiv paper)
- **URL:** https://www.open-thoughts.ai/blog/agent ; model cards https://huggingface.co/open-thoughts/OpenThinker-Agent-v1 and https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT ; dataset cards https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT and https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL
- **Source type:** official blog (with model/dataset cards)
- **Relevant topics:** agentic SFT, teacher selection, task-source ablation, terminal agents, RL task filtering, dev-set design, distillation

## Summary
The post announces OpenThoughts-Agent, an open project to curate datasets for training agents, and releases OpenThinker-Agent-v1, an SFT-only model, SFT traces, RL environments, two benchmarks (OpenThoughts-TBLite and Dev Set v1), and code. OpenThinker-Agent-v1 is trained in two stages: SFT on teacher traces, then RL. A task is an instruction (markdown), an environment (Dockerfile), and a pytest verifier, following Harbor; the verifier is not needed for SFT. The team ablated task sources and teachers by training students and scoring them on OpenThoughts-TB-Dev, a new 70-task set that is easier than Terminal-Bench 2.0. The authors report that RL on top of SFT gives a small improvement.

## Key Contributions
- Released SFT set of about 15,200 traces from two sources: nl2bash (synthetic shell-command tasks) and InferredBugs (C# and Java bugs collected by Microsoft, converted to tasks) (model card "Data"; blog says "approximately 15,000").
- A task-source ablation: about 10,000 tasks per source, each solved once by GPT-5-Nano, one student per source, scored on the dev set (blog; source chart).
- A teacher ablation on the Freelancer source (teacher chart).
- An RL task set of about 720 tasks (728 rows in the dataset metadata) with a three-stage filter (RL dataset card).
- OpenThoughts-TB-Dev: 70 terminal-agent tasks that the authors say "strongly correlates" with Terminal-Bench 2.0; no correlation value is given (blog "Evaluation").

## Key Figures/Tables to Study
- Source ablation chart "Model Performance on Dev Set" (image ota_source.png in the post); teacher chart "Freelancer Dataset: Teacher Model Comparison" (ota_teacher_graph.png); model-card results table. The results table in the blog body renders empty; its numbers are taken from the model card.

## Technical Details
**Task-source ablation (blog "SFT Data - Instruction Sourcing"; chart values read from bar labels)**
- The text says 15 approaches were ablated, from existing sources (Nemo, SWESmith, Mind2Web) and new ones (StackExchange Overflow, Freelancer, Taskmaster). The chart shows 21 source bars plus the starting model without agent SFT, labeled "Qwen3-8B (Base)", at 5.7.
- Highest: Stackexchange-Overflow 12.8, Synatra 11.3, Freelancer-Projects 9.8, Nemo-Prism-Math 9.8, Swesmith 9.7, Mind2Web 9.4, Code-Feedback 9.2, Taskmaster2 9.2.
- Middle: Staqc 8.2, Defects4J 7.9, Stackexchange-Tezos 7.4, Inferredbugs 7.1, All-Puzzles 7.1, Swesmith + Freelancer 6.7, Codeforces 6.6, Qasper 6.6, NL2Bash 5.7, Code-Contests 5.7.
- Below the starting Qwen3-8B (5.7): Mind2Web (2) 5.3, Synatra (2) 4.9, CodeActInstruct 3.5.
- The two sources in the released SFT set, InferredBugs (7.1) and NL2Bash (5.7), are not among the top-scoring sources in this GPT-5-Nano ablation; the post does not explain why they were selected.

**Teacher ablation (blog "SFT Data - Teacher Model"; teacher chart)**
- Freelancer source, dev-set score: GLM-4.6 17.2; GPT5-Nano 8.2; GPT5 8.0; GPT5-Mini 7.6.
- Text: varying teachers in the GPT family "did not improve performance, up to and including the best model on TerminalBench itself, GPT5"; GLM-4.6 "led to almost a 2x improvement in downstream score" (RL dataset card; blog has the same sentence without the GPT5 clause).

**Released SFT traces (SFT dataset card)**
- Teacher checkpoint QuantTrio/GLM-4.6-AWQ; agent harness Terminus-2; maximum 32 turns; vLLM default sampling parameters; maximum context 64K.
- Whether traces were filtered by verifier success, and the number of traces per task, are not stated.

**RL data (blog "RL Data"; RL dataset card)**
- Seeds: human-written NL2Bash queries and bash commands. GPT-5 Mini permutes instructions and commands, writes tests, and runs them in a Daytona sandbox.
- Filter (blog): remove tasks on which GPT-5-Codex gets zero reward; about 700 tasks remain from 10,000 generated.
- Filter (card): (1) drop tasks with flaky or slow verifiers; (2) drop tasks whose containers take too long to build or tear down; optional (3) drop tasks GPT-5 Codex cannot solve in a single pass.
- RL stack: SkyRL integrated with Harbor (blog "Reinforcement Learning").

**Results (model card table; harness per row)**
- Qwen3-8B (Terminus-2): Terminal-Bench 2.0 0.0; SWE-Bench Verified 0.7; OpenThoughts-TB-Dev 5.7.
- OpenThinker-Agent-v1 (Terminus-2): 4.9; 15.7; 17.3.
- Qwen3-32B (Terminus-2): 1.9; 5.7; 10.2.
- Qwen3-Coder-30B-A3B-Instruct (OpenHands): 10.1; 49.2; 24.5.
- RL over the SFT-only model: "around ~2%" on the dev set and "1%" on SWE-Bench Verified (blog); whether these are points or relative change is not stated, and SFT-only absolute scores are not given.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenThinker-Agent-v1-SFT | 8B (base Qwen3-8B) | distill-SFT | teacher; harness; turns; sampling; context | QuantTrio/GLM-4.6-AWQ; Terminus-2; max 32 turns; vLLM defaults; 64K | HF dataset card OpenThoughts-Agent-v1-SFT (README) | verified 2026-09-14 | teacher chart (Freelancer source): GLM-4.6 17.2 vs GPT5 8.0 |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | traces; sources | ~15,200; nl2bash + InferredBugs | HF model card "Data"; blog ("approximately 15,000") | verified 2026-09-14 | source chart (GPT-5-Nano traces): Inferredbugs 7.1, NL2Bash 5.7; selection not explained |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | LR; schedule; warmup | 4e-05; cosine; warmup ratio 0.1 | HF model card OpenThinker-Agent-v1-SFT "Training hyperparameters" | verified 2026-09-14 | no ablation reported |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | batch; devices; epochs | per-device 1; 16 devices; total 16 (unit not stated); 7.0 epochs | same card | verified 2026-09-14 | no ablation reported |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | optimizer; framework | AdamW (torch fused), betas (0.9, 0.98), eps 1e-08; fork of Llama-Factory; Transformers 4.56.0 | same card; blog "SFT Training" | verified 2026-09-14 | no ablation reported |
| OpenThinker-Agent-v1 | 8B | RL | tasks | ~720 (card); 728 rows (dataset metadata); ~700 from 10,000 (blog) | HF RL dataset card; blog "RL Data" | verified 2026-09-14 | no ablation reported |
| OpenThinker-Agent-v1 | 8B | RL | framework | SkyRL + Harbor | blog "Reinforcement Learning" | verified 2026-09-14 | no ablation reported |
| OpenThinker-Agent-v1 | 8B | RL | algorithm, LR, KL, samples per task, steps, max length; SFT sequence length, loss masking, packing; compute | not reported | checked blog, both model cards, both dataset cards | not reported | — |

## Findings relevant to generality, negative feedback, agentic training, distillation
- Distillation (teacher): a teacher's own benchmark strength did not predict student gain within the GPT5 family; GLM-4.6 traces scored higher (teacher chart). **Result (single study)**: one source, one student, one dev set, number of runs not stated.
- Negative marginal value: three task sources trained a student below the starting Qwen3-8B model on the dev set (Mind2Web (2) 5.3, Synatra (2) 4.9, CodeActInstruct 3.5 vs 5.7) (source chart).
- Generality measurement: the dev set is new (70 tasks) and chosen because Terminal-Bench 2.0 gives near-zero scores for small models (blog "Evaluation"); results are also reported on SWE-Bench Verified.
- Agentic RL: the RL gain over SFT is reported as small (about 2% dev, 1% SWE-Bench Verified); the RL task filter removes tasks a strong model cannot solve, so RL uses only tasks with non-zero reward for GPT-5-Codex (blog).

## Connections
- [[open-thoughts]] — the same project's reasoning-data recipe; as there, the SFT pipeline uses a fork of Llama-Factory (blog "SFT Training").
- [[terminal-bench-2]], [[terminal-bench-trajectories]] — the target benchmark and another terminal-agent trajectory source.
- [[swe-smith]] — one of the ablated task sources (Swesmith 9.7).
- [[skyrl-agent]] — the SkyRL family used for the RL stage.
- [[distillation-source-matters]], [[capacity-gap-law-distillation]] — other evidence that teacher choice changes student quality.
- [[together-coderforge-agent-trajectories]], [[agent-data-protocol]] — other open agent-trajectory SFT datasets.
- [[glm-4-5]] — report for the GLM-4.5 predecessor of the GLM-4.6 teacher.

## Verification
- Created on 2026-09-14 from https://www.open-thoughts.ai/blog/agent (page dated 2025-12-05, HTML and both chart images), plus HF README files of OpenThinker-Agent-v1, OpenThinker-Agent-v1-SFT, OpenThoughts-Agent-v1-SFT, and OpenThoughts-Agent-v1-RL (main branch, fetched 2026-09-14).
- Audit claims not found in the source: "the card's citation metadata shows 2024" (all cards show year = {2025}); "about one trace per task" holds only for the GPT-5-Nano source ablation ("solve each task once"), not for the released GLM-4.6 SFT set, where trace count per task is not stated.
- Not reported by the source: RL algorithm and hyperparameters, SFT-only absolute scores, number of evaluation runs, dev-set correlation value, reason for the final SFT source choice.
