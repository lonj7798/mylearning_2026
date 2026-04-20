# Emergent Abilities of Large Language Models
- **Authors:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2206.07682
- **Core Insight:** Some capabilities appear to emerge suddenly at scale; or do they? (Sparked massive debate.)
- **Guideline:** Do not assume that model capabilities scale linearly. Evaluate models at multiple scales and on diverse benchmarks, because some abilities may appear to transition sharply from absent to present as scale increases.
- **Relevant chapters:** Scaling laws, Model capabilities, Evaluation, Emergent behavior

## Abstract
Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models.

## Key Contributions
- Defined "emergent abilities" formally: capabilities that are absent in smaller models but appear in larger models, and cannot be predicted by extrapolating from smaller scales
- Catalogued numerous examples of emergent abilities across different model families, tasks, and benchmarks (including BIG-Bench)
- Distinguished between emergence in few-shot prompting scenarios and emergence in augmented/chain-of-thought prompting strategies
- Argued that emergence implies additional scaling could unlock further unpredictable capabilities
- Sparked a fundamental debate about whether phase transitions in capability are real properties of models or artifacts of measurement

## Why This Paper Matters
This paper crystallized one of the most important and contested ideas in modern AI: that scaling can produce qualitative jumps in capability that are not foreseeable from smaller models. It influenced investment decisions (justifying training ever-larger models), safety discussions (unpredictable capabilities are harder to govern), and scientific methodology (how we evaluate and benchmark models). The subsequent challenge by Schaeffer et al. makes this paper even more important as a case study in how measurement choices shape our understanding of AI progress.
