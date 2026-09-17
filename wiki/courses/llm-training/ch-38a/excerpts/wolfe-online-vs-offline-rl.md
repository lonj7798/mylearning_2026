---
chapter: ch-38a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/blogs/wolfe-online-vs-offline-rl.md (planned card; not present on 2026-09-15)
source_url: https://cameronrwolfe.substack.com/p/online-rl
created_at: "2026-09-15"
---

# Excerpt: Online versus Offline RL for LLMs (Cameron R. Wolfe, Deep (Learning) Focus)

**Author/organization:** Cameron R. Wolfe (newsletter post, September 2025).
**Source type:** secondary review. The post runs no experiments of its own; every number in it is attributed to a paper. It is used in ch-38a only for the taxonomy of online, semi-online and offline training, never as evidence for a quantitative claim.
**Status:** no library card existed for this slug on 2026-09-15; read from the fetched page text on 2026-09-15.

## Taxonomy quoted
"enhancing offline algorithms with on-policy data can form semi-online algorithms that are effective and easier to implement relative to full online RL."

Semi-online DPO is described with a sync period s: "the policy being trained is used to generate fresh on-policy samples for DPO every s training iterations … By varying the setting of s, we can explore arbitrary granularities of semi-online DPO, even including a fully on-policy DPO setting where s = 1." Offline DPO is the s → ∞ end of the same axis.

## Papers the post summarizes (primary sources for any number)
- Tang et al., "Understanding the performance gap between online and offline alignment algorithms", arXiv:2405.08448 — card [[on-off-policy-rlhf]].
- Tajwar et al., arXiv:2404.14367 — excerpt [[on-policy-suboptimal-preference-data]].
- Xu et al., "Is DPO superior to PPO for LLM alignment?", arXiv:2404.10719.
- Ivison et al., "Unpacking DPO and PPO", NeurIPS 2024.
- Lanchantin et al., "Bridging Offline and Online Reinforcement Learning for LLMs", arXiv:2506.21495 — the source of the semi-online DPO experiments (Llama-3.1-8B-Instruct; WildChat-1M with an Athene-RM-8b reward for the non-verifiable domain, NuminaMath with Math-Verify for the verifiable domain). The post reports that online and semi-online DPO outperform offline DPO in both domains and that semi-online stays close to online even at s = 100; these numbers were not verified against the paper for this chapter and are cited as "reported by a secondary review, not verified here".

## Caution recorded
The post's reference numbering is inconsistent in places (the same bracket number is used for Tang et al. and Tajwar et al. in different paragraphs). Do not take attributions from this post without checking the primary source.

## How ch-38a uses it
§5 (the online / semi-online / offline axis and the sync-period parameter s), Sources.
