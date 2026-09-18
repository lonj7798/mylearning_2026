<!-- chapter excerpt for ch-54. Primary-source extract, read 2026-09-17.
     If a library card of this slug exists under wiki/raw-data/llm-training/, prefer the card. -->

# Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting
- **Artifact:** arXiv:2510.18874 (v1 2025-10; text read from v3, 2026-06-26). Source type: paper.
- **Core Insight:** Across Llama and Qwen models up to 8B and across instruction following, general knowledge, and arithmetic reasoning, RL forgets less than SFT while reaching comparable or higher target-task performance; the authors attribute the difference to the mode-seeking behavior that follows from using on-policy data (Abstract, §1, Fig. 2).
- **Guideline:** When a fine-tuning stage has a choice between an equally good on-policy and off-policy data source, prefer the on-policy one if prior capability must be retained, because in this study SFT showed severe forgetting while RL reached high target-task accuracy without substantial forgetting on the same tasks (Abstract, Fig. 2).

## Technical details (with loci)
- **Setting (§1).** Qwen 2.5 and Llama 3 models up to 8B; three task families — instruction following, general knowledge, arithmetic reasoning.
- **Result (Abstract, §1, Fig. 2).** A consistent trend across model families and tasks: SFT suffers severe forgetting; RL reaches high target-task performance without substantial forgetting.
- **Analysis (Abstract).** A simplified model treats the LM as a mixture of a prior-knowledge distribution and a target-task distribution. The mode-seeking nature of RL, which the authors trace to its use of on-policy data, is identified as what leaves prior knowledge intact; the insight is then verified in the practical setting.
- **Relation to the forward-KL argument (§1).** The authors note the apparent tension with the standard view that SFT minimizes the forward KL to the target distribution, and treat resolving it as the point of the analysis.

## Not reported
Effect of partially off-policy (stale) rollout data at a controlled staleness level; agentic or long-horizon tasks.
