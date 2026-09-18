<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Environments Hub: A Community Hub To Scale RL To Open AGI
- **Artifact:** Prime Intellect Team, announcement post, 2025-08-27, primeintellect.ai/blog/environments. Source type: official blog (announcement; no experiments). Read 2026-09-17.
- **Core Insight:** RL environments are treated as a distributable artifact with their own registry, not as code embedded in one training repository; the post launches a public hub where environments are created, shared, evaluated, and consumed by a trainer.
- **Guideline:** When an RL run needs environments outside the team's own domain, treat the environment as a versioned dependency with its own evaluation report, because that is the interface the hub exposes: environments, eval reports, and sandboxes that plug into the same verifier interface.

## Technical details (with loci)
- **What is published (§"Features").** Environments can be created, managed, and shared for both reinforcement learning and evaluation; evaluation reports for environments across models are a separate first-class object; environments are natively supported by the `prime-rl` trainer (github.com/PrimeIntellect-ai/prime-rl); sandboxes for secure code execution are launched in beta and plug into verifier environments.
- **Definition used (§"Motivation").** "Environments define the world, rules and feedback loop of state, action and reward", spanning games, coding tasks, and dialogue.
- **Adoption stated (§"Contributors").** Over 30 researchers and companies contributed environments during the private beta week before launch; contributors named include Arcee AI, Hud.so, WhyPhy Labs, and Groq.
- **Related tooling.** `verifiers` and `prime-environments` repositories; open RFCs and bounties aimed at an open INTELLECT-3 model on agentic and coding tasks.

## Not reported
No counts of environments, tasks, or training runs; no measurement of any kind. This is an announcement, so it cannot support a quantitative claim.
