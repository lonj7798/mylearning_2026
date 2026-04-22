<!-- scope: Stanford Alpaca as the low-cost Self-Instruct replication recipe
     deps: [[self-instruct]]
     see-also: [[lima]], [[self-instruct]]
-->

# Alpaca: A Strong, Replicable Instruction-Following Model
- **Core Insight:** A strong open instruct model can be built cheaply by applying a simplified Self-Instruct pipeline with a proprietary teacher over a good base model.
- **Guideline:** If you need a low-cost academic replication of instruction tuning, seed from a small instruction set, expand with a strong teacher, and fine-tune a capable open base model.
- **Authors:** Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori Hashimoto
- **Year:** 2023
- **URL:** https://crfm.stanford.edu/2023/03/13/alpaca
- **Relevant topics:** instruction synthesis, teacher distillation, low-cost SFT

## Abstract
Alpaca fine-tunes LLaMA 7B on 52K instruction-following demonstrations generated in the style of Self-Instruct using `text-davinci-003`. The release showed how cheaply a useful open instruct model could be reproduced once both a strong base model and a synthetic-data pipeline were available.

## Key Contributions
- Simplified Self-Instruct into a low-cost open reproduction recipe.
- Released a 52K synthetic instruction dataset and training code.
- Made synthetic instruction tuning accessible to academic groups.

## Technical Details
- Base model: LLaMA 7B.
- Data generation starts from the Self-Instruct seed set and expands with `text-davinci-003`.
- Reported cost is under roughly `$500` for data generation and under `$100` for fine-tuning in the original release.
- The main limitation is teacher dependency and legal/license constraints.

## Connections
- Direct descendant of [[self-instruct]].
- Important historical bridge to later open synthetic SFT ecosystems like [[wizardlm]] and [[openhermes]].
