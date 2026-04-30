<!-- scope: Moonshot AI (Kimi) contributions to LLM post-training -->

# Moonshot AI (Kimi)
- **Core Contribution:** Trillion-parameter MoE with MuonClip-stabilized pretraining, and post-training specialized for agentic long-horizon tasks via joint RLVR + self-critique rubric rewards.
- **URL:** https://kimi.ai/ | https://github.com/MoonshotAI
- **Key people:** Yang Zhilin (CEO), Zhou Xinyu, and the Kimi research team

## Signature artifacts
- [[kimi-k2]] — Kimi K2 technical report (arXiv 2507.20534); 1T-MoE with 32B active params, 15.5T pretraining tokens, MuonClip optimizer, agentic joint-RL post-training.
- Kimi k1.5 — earlier reasoning release referenced across RL literature.
- Kimi K2.6 — productization updates focused on agentic coding.
- Long-context flagship Kimi Chat (pre-K2) was among the first Chinese models to ship 200K+ context windows.

## Position in the field
Moonshot AI is the agentic-specialization lab among the frontier Chinese open releases. Where DeepSeek optimizes for reasoning (R1) and Qwen optimizes for breadth, Kimi optimizes for tool-use / multi-turn agentic behavior. The K2 technical report is the cleanest disclosure of a joint-reward RL scheme in the open — combining verifiable rewards (math, code execution, tool-call correctness) with self-generated rubric rewards that evaluate model outputs against model-produced criteria. Their signature infrastructure move is MuonClip: a Muon-optimizer variant that rescales query/key weights post-update to cap attention logit magnitude, which they show is necessary to scale Muon to trillion-parameter training without divergence.

Kimi ships one of the few 1T-total-parameter open-weight models (alongside DeepSeek-V3 at 671B), and the post-training disclosure is richer than most Chinese frontier labs.

## Anticipated future work
- Continued emphasis on agentic coding (K2.6 production release).
- Extension of self-critique rubric RL to richer multi-turn settings.
- Potential publication of full MuonClip training dynamics at larger scale.

## Related pages
- [[kimi-k2]].
- [[deepseek]] — primary frontier-Chinese competitor; contrast in optimizer and RL objective.
- [[alibaba-qwen]] — same ecosystem, broader breadth focus.
- [[self-rewarding-lm]] — self-critique rubric lineage.
