<!-- scope: OLMo 3 fully open model-flow report from pretraining to RLVR
     deps: [[dolma]], [[tulu-3]]
     see-also: [[qwen-3]], [[deepseek-v3]], [[allen-ai]]
-->

# Olmo 3
- **Core Insight:** The real scientific artifact is not just the final model weights but the entire model flow: pretraining stages, intermediate checkpoints, datasets, post-training branches, evals, and tooling.
- **Guideline:** If you want a training corpus for research rather than just deployment, study OLMo 3 as a model-flow release: broad pretraining, targeted mid-training, long-context extension, then separate SFT/DPO/RLVR branches for instruct, think, and RL-zero pathways.
- **Authors:** Team Olmo
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2512.13961
- **Relevant topics:** fully open model flow, staged training, RLVR, long-context training, transparency, AllenAI

## Abstract
OLMo 3 is a fully open 7B and 32B model family targeting long-context reasoning, function calling, coding, instruction following, general chat, and knowledge recall. The key claim is not only that the models are strong, but that every stage of their construction is public: data, checkpoints, code, and dependencies across pretraining, mid-training, long-context extension, and post-training. The result is a reusable research scaffold for studying how capabilities emerge and how training choices alter them.

## Key Contributions
- Treats the **entire model flow** as the public artifact, not just final checkpoints.
- Releases multiple branches from the same base: **Base**, **Think**, **Instruct**, and **RL Zero**.
- Uses a clear **three-stage base training recipe**: pretraining, mid-training on harder distributions, and long-context extension.
- Uses a clear **three-stage post-training recipe** inherited from Tulu 3: **SFT -> DPO -> RLVR**.
- Makes the data curriculum explicit: **Dolma 3**, **Dolma 3 Mix**, **Dolmino**, **Longmino**, and **Dolci**.
- Couples model release with tooling: **Olmo-core**, **Open Instruct**, **OLMES**, **OlmoTrace**, decontamination, and dedup utilities.

## Key Figures/Tables to Study
- **Model flow diagram:** the core conceptual contribution of the report.
- **Reasoning-model comparison table:** OLMo 3-Think 32B against Qwen 3 / DeepSeek-R1-Distill class models.
- **Expanded training data section:** rare public disclosure of the full data curriculum, not only the initial pretraining corpus.
- **Efficiency section:** concrete systems lessons for making post-training cheaper and faster.

## Technical Details

### Family structure
- Base models at **7B** and **32B**.
- Reasoning-focused **Think** models at **7B** and **32B**.
- Chat/tool-use **Instruct** path.
- **RL Zero** path for direct RL experimentation from the base model.

### Base-model training stages
1. **Initial large-scale pretraining** for broad text, code, and math coverage.
2. **Mid-training** on harder data distributions to sharpen programming, quantitative reasoning, and reading comprehension.
3. **Long-context extension** on very long documents.

### Data curriculum
- **Dolma 3:** about **9.3T** source tokens spanning web pages, science PDFs processed with `olmOCR`, code, math problems/solutions, and encyclopedic text.
- **Dolma 3 Mix:** about **5.9T (~6T)** pretraining tokens with stronger math/code emphasis and stronger decontamination.
- **Dolma 3 Dolmino:** **100B** mid-training tokens sampled from a ~2.2T high-quality pool for math, science, code, instruction following, and reading comprehension.
- **Dolma 3 Longmino:** about **50B** long-context tokens from a **639B**-token pool of long documents plus mid-training data.
- **Dolci:** post-training suite with separate mixes for **SFT**, **DPO**, and **RLVR**.

### Post-training
- Each main branch follows **SFT -> DPO -> RLVR**.
- The **Think** branch uses thinking-specific SFT, thinking DPO, and RLVR to elicit high-quality reasoning traces.
- The **RL Zero** branch exists specifically to study RLVR from the base model without hiding the intermediate path.

### Efficiency and infrastructure
- Pretraining used up to **1,024 H100 GPUs**.
- Mid-training used **128 H100 GPUs**.
- Post-training used **256 H100 GPUs**.
- Moving SFT from **Open Instruct** to **Olmo Core** reportedly improved throughput by **8x**.
- In-flight weight updates, continuous batching, and threading work made RL training about **4x** more efficient.

### Why OLMo 3 matters
- It is one of the clearest public examples that **openness can apply to training trajectories**, not only to final checkpoints.
- For a learner, it is unusually valuable because you can study where a capability was added: base, mid-training, long-context, DPO, or RLVR.

## Connections
- [[tulu-3]] supplies the post-training recipe OLMo 3 builds on.
- [[dolma]] is the earlier AllenAI data-transparency foundation; OLMo 3 extends that worldview to the whole flow.
- [[qwen-3]] and [[deepseek-v3]] are the closest open-model comparisons from 2025, but OLMo 3 is much more explicit about intermediate stages.
- [[allen-ai]] is the right higher-level summary if you want the researchers and the lab worldview behind this report.
