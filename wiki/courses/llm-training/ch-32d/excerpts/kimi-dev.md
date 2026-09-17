---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Yang, Wang, Fu, He, Xiong, Liu, et al. (Moonshot AI and others), "Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents", arXiv:2509.23045v3 (2025-12-08)
source_url: https://arxiv.org/abs/2509.23045
created_at: "2026-09-15"
source_type: paper
---

# Excerpt: Kimi-Dev — Agentless mid-training data and the skill-prior comparison

No library card for this source exists at the time of writing (planned slug `kimi-dev`). Only passages used by [[read]] §5, §9, §11, the negatives section, and the Recipe table are extracted.

## Framework dichotomy (§2.1)

- "Agentless approaches decompose SWE tasks into modular workflows ... Typical workflows consist of bug localization, bug repair, and test generation. This design provides modularity and stability: each step could be optimized separately as a single-turn problem with verifiable rewards".
- "By contrast, SWE-agents adopt an end-to-end, multi-turn reasoning paradigm ... Rather than following a fixed workflow, they iteratively plan, act, and reflect".

## Mid-training data (§3.2, App. A)

- "we perform mid-training with ∼150B tokens in high-quality and real-world data. With the Qwen 2.5-72B-Base ... model as a starting point, we collect millions of GitHub issues and PR commits to form its mid-training dataset, which consists of (i) ∼50B tokens in the form of Agentless derived from the natural diff patch, (ii) ∼20B tokens of curated PR commit packs, and (iii) ∼20B tokens of synthetic data with reasoning and agentic interaction patterns (upsampled by a factor of 4 during training)." Repositories in the SWE-bench Verified test set are excluded (§3.2); App. A keeps repositories with at least five GitHub stars and only MERGED PRs.
- Natural diff patches use the Agentless prompt templates with "a loss mask to the prompt part" (App. A).
- Synthetic reasoning data (App. A): a Qwen-2.5-72B-Instruct model lightly fine-tuned on 2,000 R1 trajectories generates localization rollouts; "We retain only the rollouts that achieve exactly correct file localizations. This procedure yields approximately ∼10B tokens".
- Synthetic agentic interactions (App. A): non-executing tools for viewing files and keyword search; Qwen-2.5-72B-Instruct produces rollouts; "We apply a loss mask only to the system prompt, and enable the model to simultaneously learn both actions and observations along the trajectory ... This approach integrates both policy and world modeling into mid training." In Stage 2 the agent is allowed to open incorrectly localized files and "we artificially inject agentic reasoning patterns such as 'I realize that I do not need to modify this file'", "exposing it to false-positive contexts". Ground-truth commit messages become reasoning steps and code updates become tool actions. This part contributes "approximately ∼10B tokens".
- Training (App. A): next-token prediction; synthetic part upsampled 4×; global batch 256; maximum sequence length 32K; LR 2e-5 with cosine decay to 2e-6; warm-up "approximately 3 billion tokens", decay "until approximately 150 billion tokens are processed".

## Mid-training token budget (§3.5.2, Figure 2)

- Subsets of 50B, 100B, and about 150B tokens, each followed by the same light SFT on 2,000 BugFixer input-output pairs: "increasing the number of tokens in mid-training consistently improves model performance". Values appear only in Figure 2.

## Prior comparison (§4.1-§4.2, App. G.4)

- SWE-Agent SFT on 5,016 SWE-smith trajectories (Claude 3.7 Sonnet) at 64K context: 48.6% pass@1 on SWE-bench Verified (Table 2).
- Four priors (Base, mid-trained MT, SFT, RL) fine-tuned on nested trajectory subsets with SFT token budgets {0, 2²¹, 2²³, 2²⁴, 1.1 × 2²⁵, 1.1 × 2²⁶, 1.1 × 2²⁷, 1.5 × 2²⁸}: "To achieve the top pass@1 performance of the Base prior, the RL prior needs only 2²³ SWE-Agent SFT tokens, whereas the Base prior consumes 1.5 × 2²⁸ tokens." "The MT prior is lagged behind the SFT and the RL ones in extremely data-scarce settings (zero-shot (0) and one-step gradient descent (2²¹)), but quickly becomes on par with them after 200 trajectories (2²⁴) are available for finetuning." (§4.2, Fig. 5)
- End-to-end RL after one SFT step of 2²¹ tokens: problems with pass@8 > 0 are 260 of 6,202 for MT and 2,062 for the SFT and RL priors; MT pass@1 fell below 2% after 10 RL steps (§4.2 text; Fig. 7 plots only the SFT-prior and RL-prior runs).
- App. G.4: on SWE-bench-Live and SWE-bench Multilingual, SFT and RL priors generalize better than Base with few adaptation trajectories; with more trajectories Base and MT approach them.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2509.23045 (v3), §3.2, §3.5.2, §4, App. A, App. G.4. The chapter-local extract in ch-31 of this course covers the RL and test-time parts of the same paper.
- Not reported by the source: numeric values behind Figures 2, 5, and 7 in text form; any evaluation outside software-engineering benchmarks.
