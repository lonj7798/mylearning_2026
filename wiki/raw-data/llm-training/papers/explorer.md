<!-- scope: multimodal web trajectory synthesis for open web agents
     deps: [[agentinstruct]]
     see-also: [[magnetic-one]], [[webarena-data]]
-->

# Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents
- **Core Insight:** Large web-agent datasets can be synthesized affordably by exploring the web first, then refining successful trajectories into reusable multimodal training traces.
- **Guideline:** For web-agent data, decouple task-intent discovery from final trajectory generation; exploration is what broadens the task distribution.
- **Authors:** Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, Yu Su, Ahmed Awadallah
- **Year:** 2025
- **URL:** https://www.microsoft.com/en-us/research/publication/explorer-scaling-exploration-driven-web-trajectory-synthesis-for-multimodal-web-agents/
- **Relevant topics:** web-agent data, trajectory synthesis, multimodal agents, exploration

## Abstract
Explorer tackles the data bottleneck for open web agents by synthesizing a very large multimodal web-trajectory dataset. The key idea is to use broad web exploration and refinement to generate diverse intents and successful trajectories at relatively low cost.

## Key Contributions
- Synthesizes a large multimodal web-trajectory corpus.
- Separates exploration from refinement to improve diversity.
- Shows that data scale is a major driver of open web-agent performance.

## Technical Details
- Public summary reports over `94K` successful multimodal web trajectories, `49K` unique URLs, and large screenshot / element collections.
- Average cost per successful trajectory is reported as low enough to make community-scale data collection plausible.
- Training signal is trajectory-level, not only final answer supervision.

## Connections
- Web-agent analogue of the data-scaling story seen in [[agentinstruct]].
- Closely related to open web benchmarks and agent-training stacks such as [[webarena-data]] and [[magnetic-one]].

