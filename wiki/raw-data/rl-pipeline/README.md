<!-- scope: README for the rl-pipeline raw source library (course: wiki/courses/rl-pipeline/outline.json)
     see-also: [[mai-rl-infrastructure]], [[mai-thinking-1]]
-->

# rl-pipeline source library

**Scope.** Sources for the `rl-pipeline` course: large-scale RL training infrastructure (async controller/worker/learner loops, policy staleness, weight transfer, low-precision rollout, sampler–learner numerical parity, fault tolerance, elasticity) and per-domain RL pipeline recipes (tasks, rollouts, graders, reward composition, stage order). The anchor is Microsoft AI's "RL infrastructure at Microsoft AI" post on Rocket ([[mai-rl-infrastructure]]).

**Layout.**

| Dir | Contents |
|---|---|
| `blogs/` | lab and research blog posts |
| `papers/` | arXiv papers and paper-like research notes |
| `model-reports/` | model technical reports (RL sections) |
| `frameworks/` | framework documentation and code-reading cards |
| `repos/` | local-only shallow clones of slime, verl, AReaL (and open-instruct) for code reading; gitignored via `.gitignore` (`repos/`) and never committed |

**Card schema.** Same header as `wiki/raw-data/llm-training/frameworks/slime.md`: an HTML comment (`scope` / `deps` / `see-also`), a title, bullets Core Insight / Guideline / Authors / Year / URL / Source type / Relevant topics, then `## Summary`, `## Key Contributions`, `## Key Figures/Tables to Study`, `## Technical Details`, `## Verification`. Each card stays under 120 lines; every number is quoted from the source with a location, and `## Verification` states how and when the source was read.

**Related library.** Cards already in `wiki/raw-data/llm-training/` are linked, not copied: [[slime]], [[verl-rollout]], [[areal-async-rl]], [[async-rollout]], [[grpo]], [[deepseek-open-infra]], [[rstar2-agent]], [[kimi-k2-agentic-data]], [[deepseek-v4-recipe]].

**Index files.** `COLLECTION-PLAN.md` (checklist of target cards), `insights.md` (one row per card: core insight and chapters served).
