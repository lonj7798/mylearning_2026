<!-- scope: OpenAI alignment research contributions to LLM post-training -->

# OpenAI Alignment
- **Core Contribution:** Codified the RLHF pipeline (InstructGPT), defined the reward-model-overoptimization problem (Goodhart in RM), and initiated the weak-to-strong-generalization research program.
- **URL:** https://openai.com/research | https://openai.com/safety
- **Key people (historically):** John Schulman, Jan Leike (until 2024), Ilya Sutskever (until 2024), Paul Christiano (former), Long Ouyang, Nisan Stiennon, Ryan Lowe. Superalignment team disbanded 2024; many alumni now at Anthropic / Safe Superintelligence / independent.

## Signature artifacts
- [[rlhf-instructgpt]] — Ouyang 2022; the canonical three-stage RLHF pipeline (SFT -> RM -> PPO).
- [[reward-model-overoptimization]] — Gao 2022; Goodhart's law in reward-model training.
- Learning to Summarize from Human Feedback — Stiennon 2020; the earlier RLHF-on-summarization paper.
- Weak-to-Strong Generalization (Burns 2023) — Superalignment flagship paper on using weaker supervisors to elicit strong behavior from stronger models.
- PPO (Schulman 2017) — pre-LLM but foundational.
- John Schulman's KL-approximation blog (k1/k2/k3 estimators).
- GPT-4 + o1 + o3 system cards (post-training at highest level of disclosure).

## Position in the field
OpenAI's alignment team (the original one, pre-2024 reorg) produced the InstructGPT paper that set the RLHF template for the entire field. Their angle has historically been: scale RL techniques from classic RL (PPO, AC) to language-model fine-tuning, then layer reward-model training on top. The reward-model-overoptimization paper is still the reference citation for Goodhart's law in LLM RL.

The 2023 Weak-to-Strong Generalization paper launched a new research line — using smaller / weaker models to supervise larger / more capable ones — as a proposed scalable alignment technique. This program largely dispersed with the dissolution of the Superalignment team in 2024; many of its researchers moved to Anthropic, SSI, or independent labs.

Production models (o1, o3, GPT-5) ship with system cards disclosing high-level training methodology (chain-of-thought RL, reasoning distillation, refusal training) but not hyperparameters, making the public signal thinner than Meta, Allen AI, or DeepSeek.

## Anticipated future work
- o-series reasoning models continue to evolve; post-training disclosures remain sparse.
- Alumni of the Superalignment team (at Anthropic and elsewhere) continue the weak-to-strong research lineage.
- Schulman's independent / Thinking Machines work continues the RL-for-LLM algorithmic thread.

## Related pages
- [[rlhf-instructgpt]], [[reward-model-overoptimization]], [[ppo]].
- [[john-schulman-kl-tricks]] — Schulman's blog on KL estimators.
- [[anthropic-safety-research]] — parallel (now larger) alignment research lab with overlapping alumni.
- [[constitutional-ai]] — Anthropic's alternative to RLHF that OpenAI alumni now engage with.
