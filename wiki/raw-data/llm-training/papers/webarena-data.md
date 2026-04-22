<!-- scope: agentic trajectory synthesis — WebArena + VisualWebArena trajectory collection
     deps: [[agenttuning]]
     see-also: [[openhands-data]], [[agentinstruct]]
-->

# WebArena Trajectory Collection
- **Core Insight:** Realistic web-agent training data requires a **fully reproducible, self-hosted web environment** running real apps (GitLab, Reddit, Shopping, OSM, Calendar) with deterministic state; the WebArena / VisualWebArena environments provide this substrate, and the community has since used GPT-4 rollouts in them as the source of multi-step web-agent SFT trajectories.
- **Guideline:** When training a web agent, run rollouts inside WebArena's self-hosted apps (not live sites) so trajectories are reproducible, contamination-safe, and can be re-executed for verification; gold trajectories come from GPT-4 rollouts filtered by task-success detector.
- **Authors (WebArena):** Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig (CMU + IST + …)
- **Year:** 2023 (WebArena), 2024 (VisualWebArena)
- **URL:** https://arxiv.org/abs/2307.13854 (WebArena) ; https://arxiv.org/abs/2401.13649 (VisualWebArena)
- **Relevant topics:** web agents, reproducible environments, trajectory synthesis

## Abstract
WebArena is a self-hosted, reproducible web environment consisting of five fully functional open-source applications (GitLab, Reddit-clone, Shopping, OpenStreetMap, Calendar) and a 812-task benchmark. VisualWebArena extends this with visual-grounding tasks across three image-rich apps. The environments have become the community standard for web-agent research and trajectory-collection; SFT datasets built on them include OpenHands-web trajectories and several 2024/2025 academic collections.

## Key Contributions
- **Self-hosted web env** — 5 real apps, Docker-packaged, full state reset between tasks.
- **812-task benchmark** spanning retrieval, browsing, form-filling, multi-step transactions.
- **Evaluation harness** — success checks via URL + page-state assertions.
- De facto standard for academic web-agent research.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Environment:** Docker-compose bundle with GitLab / Postmill (Reddit-clone) / Magento (Shopping) / OSM / Calendar + a deterministic initial DB state; every task has a reset script.
- **Task definition:** natural-language instruction + success criteria expressed as predicates over final URL, page content, or DOM state.
- **Trajectory collection (community approach):**
  1. Run GPT-4-based web agent (SeeAct / Webarena agent scaffold) on each task.
  2. At the end, run the success predicate.
  3. Keep trajectories that satisfy the predicate.
- **Filtering:**
  - Success predicate pass.
  - Turn count below budget.
  - No loops / no stuck states.
- **Output shape:** trajectories of 5–25 steps; each step = (observation=accessibility tree or screenshot, thought, action=click/type/scroll). Total token length 10K–50K per trajectory.
- **Teacher model:** GPT-4 (vision models for VisualWebArena); community has also used Claude-3.5 and Gemini.
- **Cost:** ~$5–10 per task rollout in GPT-4V API; dataset-scale collections run to tens of thousands of dollars.

## Modality-specific technical details (REQUIRED — agentic)
- **Environment:** 5 apps in WebArena; VisualWebArena adds Classifieds, Shopping(-visual), Reddit-visual.
- **Action space:**
  - `click [element_id]`, `type [element_id] [text]`, `hover`, `press`, `scroll`, `tab`, `new_tab`, `goto`, `go_back`, `stop [answer]`.
  - Actions grounded on accessibility tree (WebArena) or image coordinates (VisualWebArena).
- **Trajectory length:** avg 12 steps, max 30.
- **Success criterion:** predicate-based. Three categories: info-lookup, content-producing-task, state-modifying-task.
- **Data scale:** WebArena benchmark = 812 tasks; trajectory datasets built on top = tens of thousands.
- **Observation modes:**
  - **Accessibility tree (AX):** text representation of DOM, preferred for text-only agents.
  - **Screenshot + AX:** multimodal mode, used in VisualWebArena.

## Quality / diversity evaluation
- GPT-4 with best scaffold: ~35% WebArena, ~20% VisualWebArena.
- Open agents fine-tuned on WebArena trajectories: LLaVA-34B-WebArena ~15%, Claude-3-SeeAct ~25% — clear gap to frontier closed models.
- As of 2025, best open agent on WebArena ~30%, frontier closed models ~50%.

## Risks + gotchas
- **Environment drift:** Docker images must be pinned; app versions upgrading silently break tasks.
- **Shortcut learning:** some tasks solvable by URL-hacking; success predicate must be strict.
- **High per-rollout cost:** a full trajectory on GPT-4V can be $5–20.
- **Not production-realistic:** fixed 5 apps, no auth/captcha/anti-bot — real web is much harder.

## Connections
- Trajectory data consumed by: OpenHands ([[openhands-data]]), several 2025 web-agent fine-tunes.
- Eval sibling: Mind2Web (real-web), AgentBench web track.
- Environment ancestor of broader 2025 agent-RL envs.
