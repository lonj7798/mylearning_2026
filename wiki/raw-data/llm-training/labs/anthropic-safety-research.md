<!-- scope: Anthropic safety research contributions to LLM post-training -->

# Anthropic Safety Research
- **Core Contribution:** Constitutional AI (CAI) — replacing human harmlessness labels with AI-generated critiques against a principle set; foundational weak-to-strong supervision research; alignment interpretability.
- **URL:** https://www.anthropic.com/research
- **Key people:** Jared Kaplan, Yuntao Bai, Ethan Perez, Evan Hubinger, Chris Olah, Sam Bowman, Deep Ganguli, Jan Leike (joined 2024)

## Signature artifacts
- [[constitutional-ai]] — Bai 2022; RLAIF via Constitutional AI.
- Anthropic HH RLHF paper — Bai 2022 RLHF paper documenting helpful+harmless preference collection at Anthropic scale.
- Sleeper Agents (Hubinger 2024) — demonstration that backdoors survive safety training.
- Weak-to-Strong Generalization — Anthropic replications and follow-ups to the OpenAI Superalignment result.
- Model Organisms of Misalignment — red-team / safety evaluation methodology.
- Claude 3.5 / 3.7 / Opus 4 / Opus 4.5 / Opus 4.7 — production RLHF + CAI integration.
- Mechanistic interpretability lineage (Olah group): monosemantic features, sparse autoencoders, circuit analysis.

## Position in the field
Anthropic sits at the intersection of production frontier-model training and long-horizon safety research. Their signature technical move is Constitutional AI: human labels are replaced with AI-generated critiques against a written principle set, then RLHF (specifically RLAIF — RL from AI Feedback) proceeds on those AI labels. This lets harmlessness training scale without linearly more human red-teamers.

A second signature contribution is sleeper-agent research: demonstrating that if a model is trained with a backdoor trigger during pretraining, standard safety fine-tuning fails to remove it. This has been influential on how the field thinks about training-data poisoning and alignment robustness.

Anthropic does not publish technical reports on their flagship Claude models' full post-training recipes (specific SFT/RLHF/CAI data volumes, hyperparameters), but their research papers on CAI and weak-to-strong generalization are foundational. Their interpretability work (Olah group) is also a major contribution to understanding why post-trained models behave as they do.

## Anticipated future work
- Scaling supervision research (weak-to-strong generalization follow-ups).
- Continued sleeper-agent / adversarial robustness lineage.
- Alignment interpretability feeding back into training (SAE-informed training signals).
- Claude-series model cards typically disclose high-level safety methodology but not training hyperparameters.

## Related pages
- [[constitutional-ai]], Anthropic HH RLHF.
- [[openai-alignment]] — parallel lab on superalignment / weak-to-strong generalization.
- [[kimi-k2]] — self-critique rubric reward is a descendant of the CAI self-critique idea.
