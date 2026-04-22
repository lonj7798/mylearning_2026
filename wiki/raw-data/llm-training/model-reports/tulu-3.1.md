<!-- scope: Tulu 3.1 checkpoint update documented in Hugging Face model card, grounded in Tulu 3 paper
     deps: [[tulu-3]], [[grpo]], [[rlvr-tulu3]]
     see-also: [[olmo-3]], [[deepseek-r1]]
-->

# Tulu 3.1
- **Core Insight:** Tulu 3.1 is not a new end-to-end model family; it is a controlled post-training delta showing that swapping only the final RL stage from PPO to GRPO can move an 8B instruct model noticeably.
- **Guideline:** Keep the rest of the stack fixed and treat the final RL stage as an ablation target; Tulu 3.1 is valuable precisely because it isolates that change.
- **Authors / Lab:** Allen AI
- **Year:** 2025
- **URL:** https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B
- **Relevant topics:** RLVR, PPO vs GRPO, open post-training ablation, model-card-only release

## Abstract
`Tulu 3.1` is the naming Allen AI uses for an updated 8B Tulu checkpoint built from `Llama-3.1-Tulu-3-8B-DPO`. Unlike `Tulu 3`, which has a full paper describing the three-stage `SFT -> DPO -> RLVR` recipe, Tulu 3.1 is documented mainly through its Hugging Face model card. The key disclosed change is narrow and useful: Allen AI replaced the final PPO-based RL stage with **GRPO**, removed the reward model from that stage, tuned the RL hyperparameters further, and reported broad performance gains.

## Key Contributions
- A rare public example of a **single-stage post-training update** rather than a whole new family.
- The model card explicitly says the **only change** is in the **final RL stage**.
- Allen AI says it **switched from PPO to GRPO**, with **no reward model** in that final stage.
- The update keeps the rest of the Tulu 3 recipe anchored to the published **Tulu 3** pipeline.
- This makes it a clean public artifact for studying **PPO vs GRPO** in open post-training.

## Key Figures/Tables to Study
- The model-card benchmark table is the key source because it compares **Tulu 3 SFT**, **Tulu 3 DPO**, **Tulu 3**, and **Tulu 3.1** side by side.
- The Tulu 3 paper remains the right source for the earlier **SFT** and **DPO** stages.

## Technical Details

### Naming / scope note
- There is **no standalone Tulu 3.1 paper** in the sources I used.
- Publicly, `Tulu 3.1` refers to an **updated 8B checkpoint**, not to a fully re-reported family-wide recipe.

### Base lineage
- Base model: **Llama 3.1 8B**.
- Immediate parent checkpoint: **`allenai/Llama-3.1-Tulu-3-8B-DPO`**.

### What changed
- Allen AI says the new version comes from an improvement **only in the final RL stage of training**.
- The final stage switched from **PPO** to **GRPO**.
- The model card also states **no reward model** is used in that final stage.
- Additional **hyperparameter tuning** in the RL stage produced better average results than the original 8B Tulu 3 checkpoint.

### What stayed the same
- Earlier **SFT** and **DPO** stages remain those of **Tulu 3**.
- The model card still frames the data as a mix of **publicly available, synthetic, and human-created datasets**.
- The associated training dataset shown in the card is **`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`**.

### Why it matters
- Tulu 3.1 is useful because it is a **controlled public ablation** rather than a broad marketing release.
- It also shows that the open ecosystem quickly incorporated the **GRPO / verifier-style RL** trend after DeepSeek-R1 and DeepSeekMath made it prominent.

## Connections
- [[tulu-3]] is the full underlying recipe and should be read first.
- [[grpo]] is the algorithmic lens for the key update.
- [[rlvr-tulu3]] is the task-verifier RL perspective Allen AI had already been exploring.
- [[deepseek-r1]] is the nearby public model family that helped normalize GRPO-style reasoning RL.
