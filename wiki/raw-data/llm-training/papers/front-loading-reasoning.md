<!-- scope: timing of reasoning-data injection across pretraining and post-training
     deps: [[dolma]], [[self-instruct]]
     see-also: [[transferability-of-llm-reasoning]], [[echo-chamber-rl-post-training]], [[quiet-star]]
-->

# Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
- **Core Insight:** Reasoning data is most powerful when injected early: broad, diverse reasoning during pretraining creates a durable foundation that later SFT cannot fully reconstruct, even with more post-training data.
- **Guideline:** Allocate reasoning data asymmetrically across stages: maximize **diversity** during pretraining and maximize **quality** during post-training; do not assume late-stage SFT can compensate for a weak pretraining substrate.
- **Authors:** Syeda Nahida Akter, Shrimai Prabhumoye, Eric Nyberg, Mostofa Patwary, Mohammad Shoeybi, Yejin Choi, Bryan Catanzaro
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2510.03264
- **Relevant topics:** reasoning data timing, pretraining vs post-training, diversity vs quality, data curriculum

## Abstract
This paper studies how reasoning data of different scales, quality, and diversity should be introduced across pretraining and post-training. The central result is that front-loading reasoning data into pretraining yields a durable advantage that later SFT cannot fully catch up to. The paper further argues for an asymmetric allocation rule: pretraining benefits most from diverse reasoning patterns, while post-training benefits most from high-quality reasoning data.

## Key Contributions
- Systematic study of **when** reasoning data should enter the pipeline.
- Shows that front-loading reasoning into pretraining gives a reported **19% average gain** and sets a higher final ceiling.
- Identifies an asymmetric allocation principle:
  - **Pretraining:** prioritize diversity
  - **Post-training / SFT:** prioritize quality
- Refutes the idea that late-stage SFT can fully repair a weak pretraining foundation.

## Key Figures/Tables to Study
- **Headline comparison of early vs late reasoning-data injection:** core empirical takeaway.
- **Diversity-vs-quality ablations:** this is the most practically useful part for data budgeting.
- **Stage-by-stage performance growth:** useful for seeing the compounding effect of front-loaded data.

## Technical Details

### Main conclusion
- Reasoning data in pretraining creates a **durable, compounding advantage** that widens through later post-training stages.

### Asymmetric data-allocation rule
- During **pretraining**, broader diversity of reasoning patterns matters more.
- During **post-training**, especially SFT, higher-quality reasoning data matters more.

### Why this matters
- It connects directly to the 2025 RL mechanism debate:
  - if RL mostly amplifies existing priors,
  - then what priors you install during pretraining becomes crucial.

### Practical implication
- If compute is limited, do not spend all reasoning budget at the SFT stage.
- Build some reasoning structure into the base model early, then use later post-training to refine style, correctness, and task alignment.

## Connections
- [[echo-chamber-rl-post-training]] makes the strongest downstream case for why pretraining priors matter.
- [[transferability-of-llm-reasoning]] complements this on the post-training side by showing that narrow math SFT can harm generality.
- [[quiet-star]] is a conceptual cousin because it also tries to move reasoning into a pretraining-like phase rather than leaving it entirely to post-training.
- [[dolma]] and [[olmo-3]] are strong open examples of stage-aware data curricula.
