<!-- scope: Magentic-One, an inference-time multi-agent system (Orchestrator + 4 worker agents) evaluated on GAIA, AssistantBench, WebArena; no model training
     see-also: [[webarena-data]], [[explorer]], [[terminal-bench-trajectories]], [[agentinstruct]]
-->

# Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks
- **Core Insight:** An Orchestrator agent that keeps a task ledger and a progress ledger and delegates to four worker agents reaches 38.00% on the GAIA test set and 27.7% accuracy on the AssistantBench test set with GPT-4o + o1-preview, and 32.8% on all 812 WebArena tasks with GPT-4o only; the authors report the GAIA and AssistantBench results as statistically comparable to the leaderboard state of the art (Table 1, §5.2).
- **Guideline:** When one agent system must handle web, file, and code tasks with a single configuration, keep explicit planning and progress-tracking state in the orchestrator, because replacing the ledger-based Orchestrator with AutoGen's plain GroupChat speaker selection reduced GAIA-validation performance by 31% (§5.3, Fig. 3a; GPT-4o, one configuration, unit of the drop not stated).
- **Authors:** Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang (Eric) Zhu, et al. (Microsoft Research AI Frontiers)
- **Year:** 2024 (arXiv v1 2024-11; only version)
- **URL:** https://arxiv.org/abs/2411.04468
- **Source type:** paper
- **Relevant topics:** multi-agent orchestration, agent evaluation, GAIA, AssistantBench, WebArena, agent error analysis, agent safety

## Abstract
The paper introduces Magentic-One, an open-source multi-agent system for complex tasks. A task is complex if it needs planning, acting, observing, and reflecting, possibly several times (§3). A lead agent, the Orchestrator, plans, tracks progress, re-plans after errors, and directs specialized agents that operate a web browser, navigate local files, or write and execute Python code. The system reaches results statistically competitive with the state of the art on GAIA, AssistantBench, and WebArena without changing agent capabilities or collaboration between benchmarks. Agents can be added or removed without additional prompt tuning or training. The authors also release AutoGenBench, a tool that runs agentic benchmarks with repetition and isolation, and report ablations and an error analysis.

## Key Contributions
- A five-agent team (Orchestrator, WebSurfer, FileSurfer, Coder, ComputerTerminal), open-sourced and implemented on AutoGen version 0.4 (§1, §5.1).
- Ledger-based orchestration: an outer loop over a task ledger and an inner loop over a progress ledger, with a stall counter that triggers re-planning (§4.1, Fig. 2).
- AutoGenBench: each task starts in freshly initialized Docker containers, logs are stored outside the containers, and tasks can run in parallel or be repeated (§5.1).
- Benchmark results (GAIA and AssistantBench test sets, all WebArena tasks) with 95% Wald intervals and z-tests (α = 0.05) against leaderboard baselines (Table 1, App. A).
- Agent-removal ablations on GAIA validation and an LLM-automated error analysis with a code book (§5.3, §5.4, App. C).

## Key Figures/Tables to Study
- **Figure 2** — the two loops, the contents of both ledgers, and the stall-count branch.
- **Table 1** — completion rates for both configurations and all leaderboard baselines, with error bars.
- **Table 2** — results split by GAIA level, AssistantBench difficulty, and WebArena site.
- **Figure 3** — ablations by difficulty level (a) and by required capability (b).
- **Figure 4 / Appendix C** — frequency and definitions of the automatically derived error codes.

## Technical Details
- **Task ledger (outer loop):** the Orchestrator fills in given or verified facts, facts to look up, facts to derive, and educated guesses, then writes a natural-language plan that assigns steps to agents; the plan is a hint in the manner of chain-of-thought prompting, not a fixed program (§4.1).
- **Context reset:** all agents clear their contexts and reset state after each plan update (§4.1).
- **Progress ledger (inner loop):** each iteration answers five questions: is the request satisfied; is the team looping; is progress being made; which agent speaks next; what instruction to give it (§4.1).
- **Stall counter:** incremented on a detected loop or no progress; while it stays at or below the threshold (≤ 2 in the experiments) the inner loop continues; above it, the Orchestrator reflects, updates the task ledger, revises the plan, and restarts the inner loop (§4.1).
- **Termination:** task completion or configurable limits such as maximum attempts or maximum time; the Orchestrator then reports a final answer or its best educated guess (§4.1). The problem setup gives 25 minutes as an example time budget (§3).
- **WebSurfer:** LLM agent controlling a Chromium browser; maps each request to one action (navigation, clicking/typing, reading actions such as summarizing) and returns a screenshot plus a written description; grounding uses set-of-marks prompting extended with text about content outside the viewport (§4.2).
- **FileSurfer:** commands a read-only, markdown-based file preview application covering PDFs, Office documents, images, videos, and audio; can list and navigate directories (§4.2).
- **Coder:** LLM agent specialized by system prompt; writes a new standalone Python program for each request (§4.2, §6.2).
- **ComputerTerminal:** deterministic, no LLM calls; runs Python programs and shell commands, including library installs (§4.2).
- **Models:** gpt-4o-2024-05-13 for all LLM agents; a second configuration uses o1-preview for the Orchestrator's outer loop and the Coder (§5.1). Experiments ran August–October 2024 (§5.1).
- **Per-benchmark setup:** one agent configuration for all benchmarks; only the final-answer prompt, WebArena login code, and a prompt explaining that the WebArena forum is Postmill differ (§5.1).
- **Benchmarks:** GAIA 465 questions (165 validation, 300 test); AssistantBench 214 (33 validation, 181 test); WebArena 812 tasks, split by the authors via the MD5 hash of the template id into 422 validation and 390 test tasks (§5.1).
- **Results (Table 1; GAIA and AssistantBench test sets, all 812 WebArena tasks):** GPT-4o: GAIA 32.33±5.3, AssistantBench EM 11.0±4.6 and accuracy 25.3±6.3, WebArena 32.8±3.2. GPT-4o + o1-preview: GAIA 38.00±5.5, AssistantBench EM 13.3±4.9 and accuracy 27.7±6.5. Highest baselines: GAIA omne v0.1 40.53±5.6; WebArena Jace.AI 57.1±3.4 and WebPilot 37.2±3.3, both statistically higher than Magentic-One (§5.2). Human: GAIA 92.00±3.1, WebArena 78.2±2.8.
- **o1-preview refusals:** WebArena is not reported for the o1 configuration because o1 refused 26% of GitLab tasks and 12% of Shopping Administration tasks (Table 1 caption).
- **By difficulty (Table 2, GPT-4o + o1):** GAIA Level 1/2/3 = 54.84/32.7/22.92 vs best baseline 53.76/37.11/26.53; AssistantBench Hard = 14.8 (GPT-4o: 16.9) vs 13.3.
- **Ablations (GAIA validation, GPT-4o, §5.3):** removing the ledgers (simple GroupChat orchestrator) drops performance by 31%; removing agents drops it by 21% (Coder and ComputerTerminal, removed together in Fig. 3) to 39% (FileSurfer).
- **Error analysis (§5.4):** GPT-4o writes a root-cause postmortem per task log, assigns free-form codes, and iteratively clusters them; 200 random logs bootstrap the code set, which is then applied to all logs. Most frequent codes: persistent-inefficient-actions, insufficient-verification-steps, inefficient-navigation-attempts.
- **Cost:** most problems need dozens of iterations and LLM calls, "perhaps several US dollars, and tens of minutes per task"; cost was not formally evaluated (§6.2).

## Findings relevant to generality, agentic training
- **Generality across benchmarks:** the same configuration is used on all three benchmarks; among Table 1 baselines, no prior system other than base models was evaluated on all three (§5.2).
- **Validation overfitting:** on the authors' WebArena split, Magentic-One solved 35.1% (148/422) of validation tasks, used during debugging, and 30.5% (119/390) of test tasks, evaluated once; the authors interpret this as mild overfitting (§5.2, Interpretation).
- **Task length:** relative to baselines, Magentic-One does better on hard tasks than on easy ones; the authors hypothesize a fixed orchestration overhead that helps long tasks and adds error opportunities on short ones (§5.2, Interpretation).
- **Capability compensation:** with Coder and ComputerTerminal removed, agents had FileSurfer read code and predict its output; with FileSurfer removed, agents looked for an online PDF viewer (§5.3).
- **Agentic training:** no model is trained and no trajectory dataset is released. Listed limitations include no learning across tasks (§6.2). Suggested future work: smaller local models for WebSurfer/FileSurfer grounding (§6.1, §6.2) and phishing-rejection examples in post-training (§6.3).
- **Observed unsafe behaviour (§6.3):** repeated login attempts until an account was suspended, then a password-reset attempt; attempts to go to the real Reddit (blocked by network rules); accepting cookie agreements and terms without a human; attempts to recruit humans (social-media posts, emailing textbook authors, drafting a freedom-of-information request).

## Connections
- [[webarena-data]] — WebArena is one of the three evaluation benchmarks here (812 tasks, §5.1).
- [[explorer]] — synthesizes web-agent training trajectories; Magentic-One is an inference-time system with no training data.
- [[terminal-bench-trajectories]] — released agent traces used as data; in this paper, Magentic-One logs are used for error analysis (§5.4).
- [[agentinstruct]] — a separate Microsoft Research work that uses multi-agent flows to generate instruction data rather than to solve tasks.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2411.04468 (arXiv v1, 2024-11-07; full text including Appendices A–C).
- Slug note: the file name `magnetic-one` misspells the artifact name "Magentic-One" (a combination of "multi" and "agentic", §1 footnote 1). The file name is kept so links resolve.
- Corrections to the previous card version:
  - "Microsoft AutoGen team" → affiliation is Microsoft Research AI Frontiers (title page); author list shortened to first six + et al.
  - "leading open scores on GAIA and Assistant-Bench at release" / "leading complex-task success rates" → statistically comparable to SOTA on GAIA and AssistantBench; omne v0.1 is numerically higher on GAIA (40.53 vs 38.00); on WebArena, WebPilot and Jace.AI are statistically higher (Table 1, §5.2).
  - "adding a new sub-agent requires re-prompting the orchestrator" → the paper states agents can be added or removed without prompt tuning or changes to other agents' prompts (Abstract, §6.1); it lists fixed team membership as a limitation (§6.2).
  - "FileSurfer ... Docker sandbox" → FileSurfer is a read-only markdown-based file preview application; Docker containers isolate each benchmark task in AutoGenBench (§4.2, §5.1).
  - "Orchestrator drafts Task Ledger (known facts, plan) ... repeat until max-step limit" → two ledgers (task and progress), a stall threshold of ≤ 2, and termination by task completion or configurable max attempts/time (§4.1).
  - "Teacher model: GPT-4o (primary), Claude-3.5-Sonnet (alternative)" → configurations use gpt-4o-2024-05-13 and o1-preview; no Claude configuration (§5.1).
  - "running four LLM agents per step" → ComputerTerminal makes no LLM calls (§4.2).
- Removed as unsupported by the source: the whole "trajectory synthesis for data purposes" section (running the system to collect and success-filter training traces); "~10K–50K tokens per trajectory"; "several thousand successful trajectories"; "~$5–$20 per trajectory"; "median ~20 steps, tail to 100+"; "smaller agents fine-tuned on Magentic-One trajectories inherit multi-agent coordination patterns"; "Task Ledger tokens teach state tracking"; "orchestrator-agent dependency cycle"; deps link to [[agentinstruct]] (not a prerequisite).
- Not reported by the source: any model training or fine-tuning; a trajectory dataset; tokens or steps per task; per-task cost measurements.
