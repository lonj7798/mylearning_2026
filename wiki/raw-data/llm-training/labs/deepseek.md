<!-- scope: DeepSeek AI contributions to LLM post-training -->

# DeepSeek AI
- **Core Contribution:** GRPO and rule-reward RL at frontier scale — DeepSeekMath birthed GRPO, V3 productionized MoE+FP8, R1 proved pure-RL reasoning works.
- **URL:** https://www.deepseek.com/ | https://github.com/deepseek-ai
- **Key people:** Daya Guo, Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Y.K. Li, Y. Wu, and the DeepSeek research team (under High-Flyer Capital Management).

## Signature artifacts
- [[deepseekmath]] — Shao 2024; introduced Group Relative Policy Optimization (GRPO).
- DeepSeek-V2 — V2 with Multi-head Latent Attention (MLA) and MoE.
- [[deepseek-v3]] — 671B MoE with auxiliary-loss-free routing, FP8 training, 14.8T tokens, ~$5.6M claimed training cost.
- [[deepseek-r1]] — R1 and R1-Zero; first public proof that pure RL on a base model causes reasoning to emerge.
- DeepSeek-Coder — code pretraining line.
- Distilled R1 series (Qwen-1.5B/7B/14B/32B and Llama-8B/70B students).

## Position in the field
DeepSeek is the post-training lab to watch for cost-efficient, reasoning-focused open releases. Their signature move is algorithmic: GRPO (published in the DeepSeekMath paper) removed the PPO critic, halving the memory footprint of RLHF; R1-Zero then showed that rule-based rewards alone — accuracy + format — are sufficient to induce emergent long chain-of-thought. The R1 release sent the field's attention back to RL as the primary engine of reasoning after a year of DPO dominance.

Their angle is infrastructure-first: every report discloses token counts, GPU-hour budgets, stability anecdotes (no loss spikes, no rollbacks), and exact RL hyperparameters in a way most frontier labs do not. They treat post-training as a co-designed problem with pretraining — V3 is co-engineered with R1 distillation in mind, and MLA is co-engineered with long-context RL in mind.

## Anticipated future work
- V4 / R2 expected with scaled GRPO (larger group sizes, longer rollouts).
- Expansion of rule-reward RL to non-verifiable domains (agentic, dialogue) is an open question their recent papers hint at.
- Continued release cadence of distilled reasoning students targeting consumer hardware.

## Related pages
- [[grpo]], [[deepseekmath]], [[deepseek-v3]], [[deepseek-r1]].
- [[kimi-k2]] — parallel Chinese frontier lab; contrast in optimizer and reward design.
- [[tulu-3]] — RLVR is the open analogue to DeepSeek's rule-reward RL.
