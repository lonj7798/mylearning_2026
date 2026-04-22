<!-- scope: full agent trajectories from Terminal-Bench as an open terminal-agent data source
     see-also: [[openhands-data]], [[magnetic-one]]
-->

# Terminal-Bench Trajectories
- **Core Insight:** A hard terminal benchmark becomes much more valuable for training once the full step-by-step agent traces are released, not just the success rates.
- **Guideline:** For CLI agents, keep full trajectory logs with tool calls, messages, and observations; benchmark traces can double as high-value post-training data.
- **Authors:** Terminal-Bench ecosystem / trajectory dataset maintainers
- **Year:** 2026
- **URL:** https://huggingface.co/datasets/yoonholee/terminalbench-trajectories
- **Relevant topics:** terminal agents, trajectory data, CLI tasks, agent post-training

## Abstract
Terminal-Bench 2.0 trajectories release full agent traces over hard terminal tasks. This turns a benchmark into a reusable trajectory corpus for studying agent behavior, failure modes, and post-training on real CLI interaction structure.

## Key Contributions
- Makes full terminal-agent traces public instead of only leaderboard scores.
- Covers many agent/model combinations over a shared task set.
- Provides a reusable corpus of terminal interactions, tool calls, and outcomes.

## Technical Details
- Dataset card reports tens of thousands of trajectories over the Terminal-Bench task set.
- Each row stores a full trial including messages, tool calls, and observations.
- Valuable for imitation, trajectory filtering, and agent-behavior analysis.

## Connections
- Natural complement to the Terminal-Bench benchmark itself.
- Related to [[openhands-data]] and other agent-trajectory corpora.
