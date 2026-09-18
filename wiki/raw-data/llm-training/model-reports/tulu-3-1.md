<!-- scope: redirect — duplicate card for the Tülu 3.1 artifact; canonical card is [[tulu-3.1]]
     deps: [[tulu-3]]
     see-also: [[tulu-3.1]], [[grpo]], [[rlvr-tulu3]]
-->

# Tülu 3.1 (redirect)

> **Redirect card.** This file and `model-reports/tulu-3.1.md` described the same artifact. The canonical
> card is **[[tulu-3.1]]**, which is built on the primary source (the Hugging Face model card for
> `allenai/Llama-3.1-Tulu-3.1-8B`). This file is kept only so existing `[[tulu-3-1]]` links resolve.
> Cite [[tulu-3.1]] for the 3.1 delta and [[tulu-3]] for the underlying recipe.

- **URL of the primary artifact:** https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B
- **Source type:** model card
- **Underlying report:** [[tulu-3]] (arXiv:2411.15124)

## What the primary source says, in one paragraph
The model card states that the 3.1 version comes from an improvement only in the final RL stage of
training: the PPO stage was replaced by GRPO with no reward model, followed by further hyperparameter
tuning. The card lists the GRPO settings for that stage, including learning rate 5e-7, a constant learning
rate schedule, 16 samples per prompt, and KL penalty coefficient β = 0.01. The parent checkpoint is
`allenai/Llama-3.1-Tulu-3-8B-DPO` and the RL training set is
`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`. The full numbers and their loci are in [[tulu-3.1]].

## Connections
- [[tulu-3.1]] — canonical card for this artifact.
- [[tulu-3]] — the report that documents the SFT and DPO stages that 3.1 leaves unchanged.
- [[grpo]] — the algorithm the final stage was switched to.
- [[rlvr-tulu3]] — the verifiable-reward RL setting both versions use.

## Verification
- Checked on 2026-09-18 against: https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B (model card) and https://arxiv.org/abs/2411.15124 (arXiv v5)
- Corrections to the previous card version:
  - "Tülu 3.1 is the propagation of the same recipe onto Llama 3.1 and OLMo 2 bases, not a new algorithm" → the model card states the only change is in the final RL stage, PPO → GRPO with no reward model, on an 8B checkpoint whose parent is `Llama-3.1-Tulu-3-8B-DPO`. The card describes no OLMo 2 variant.
  - "Year: 2024-11-22 update (refresh of Tülu 3 Nov 21)" → the model card gives no such date for 3.1; the Tülu 3 report itself is arXiv:2411.15124 (v1 2024-11). The 3.1 release date is not established by either primary source read here.
  - "URL: https://allenai.org/blog/tulu-3-technical — https://arxiv.org/abs/2411.15124" → the card mixed two artifacts; the artifact named by the slug is the Tülu 3.1 model card.
  - "KL / entropy / LR / batch / clip / group / rollouts / RL-step counts ... 3.1 refresh does not publish per-base-model hyperparameter deltas" → the 3.1 model card does publish GRPO settings, including learning rate 5e-7, constant schedule, 16 samples per prompt and β = 0.01.
  - "Re-running the recipe on updated bases: Llama-3.1-Tulu-3-8B, -70B, -405B" → those are the Tülu 3 checkpoints, not Tülu 3.1 checkpoints.
  - "Reward model: trained RM for DPO-pair scoring" → DPO preference labels in the Tülu 3 report come from LLM-judge pairwise comparisons (§5.2.1, Table 17), and the 3.1 final stage uses no reward model at all.
- Removed as unsupported by the source: the claim that Tülu 3.1 joined OLMo 2 bases; the claim that the pipeline is "unchanged"; the DR Tulu and OLMo 3 paragraphs (they describe other artifacts and belong on their own cards); "the most fully-reproducible open recipe"; the "Key Figures/Tables to Study" entries, which pointed at the Tülu 3 paper rather than at this artifact.
- Not reported by the source: benchmark deltas for any size other than 8B; whether the SFT or DPO data changed for 3.1.
