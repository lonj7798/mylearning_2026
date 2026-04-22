<!-- scope: Yi family report from 01.AI with base, chat, long-context, and multimodal extensions
     deps: [[fineweb]], [[self-instruct]]
     see-also: [[qwen-2.5]], [[llama-3]], [[phi-3]]
-->

# Yi: Open Foundation Models by 01.AI
- **Core Insight:** Yi argues that strong open bilingual models can be built more from data engineering discipline than from exotic architecture changes.
- **Guideline:** Before inventing new architecture, make the pretraining pipeline sharper: deduplication, quality filtering, bilingual balance, and carefully hand-polished instruction data can still move the frontier.
- **Authors / Lab:** 01.AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2403.04652
- **Relevant topics:** bilingual pretraining, data engineering, long-context extension, small high-quality SFT, open foundation models

## Abstract
Yi is a family of open language and multimodal models centered on 6B and 34B pretrained backbones, with chat, 200K-context, depth-upscaled, and vision-language variants layered on top. The report attributes most of the model quality to data work rather than architecture novelty: a 3.1T-token English-Chinese corpus, cascaded deduplication and quality filtering, and a very small instruction dataset repeatedly refined by engineers.

## Key Contributions
- **3.1T-token** bilingual pretraining corpus with explicit emphasis on **English and Chinese**.
- A clear claim that **data quality** rather than architecture novelty drove performance.
- Shows a compact post-training philosophy: **less than 10K** instruction examples, iteratively refined and directly verified by engineers.
- Extends the family to **200K context** through lightweight continual pretraining.
- Expands from base LMs to **chat**, **long-context**, **depth-upscaled**, and **vision-language** variants inside one family.

## Key Figures/Tables to Study
- The abstract is already unusually useful: it states the full data and post-training philosophy in a few lines.
- Benchmark sections for **Yi-34B**, **Yi-34B-200K**, and **Yi-9B** show how the family was staged across scale and context.
- The GitHub release log is useful because it shows the sequence of public updates: Yi, Yi-9B, Yi-200K, then Yi-1.5.

## Technical Details

### Base family
- Built on **6B** and **34B** pretrained language models.
- The GitHub repository says the models follow the **Llama-style architecture / format** closely for ecosystem compatibility.

### Pretraining
- **3.1T tokens** of primarily **English and Chinese** text.
- The paper explicitly credits a **cascaded data deduplication and quality filtering pipeline**.
- The report frames scalable infrastructure plus classical Transformer design as sufficient when paired with better data.

### Post-training
- Finetuning uses a **small instruction dataset (<10K examples)**.
- The paper says the data was polished over **multiple iterations** and that each example was **verified directly by ML engineers**.
- Public material does not disclose a modern RLHF / DPO / PPO recipe for the original Yi report; the disclosed post-training story is mainly **high-quality supervised alignment**.

### Long-context and extensions
- Context is extended to **200K** through **lightweight continual pretraining**.
- The GitHub release notes mention an additional **5B-token long-context data mixture** for the enhanced long-context line.
- The family also includes **vision-language** models that align a vision transformer encoder to the language model space.

### Why it matters
- Yi is a good counterexample to the idea that every strong model jump needs a novel optimization algorithm.
- It is especially useful if you care about **data curation, bilinguality, and small-but-expensive instruction data** rather than very large synthetic SFT pools.

## Connections
- [[qwen-2.5]] is the closest Chinese/English open-family comparison, but with a much more synthetic and larger-scale later recipe.
- [[llama-3]] is a contrast case: much heavier iterative synthetic post-training.
- [[phi-3]] is another data-centric report, but with more explicit synthetic-textbook framing.
