<!-- scope: GLM-4 family report with note on family-level naming ambiguity
     deps: [[bfcl]], [[toolformer]]
     see-also: [[qwen-2.5]], [[yi]], [[deepseek-v3]]
-->

# GLM-4
- **Core Insight:** GLM-4 is best understood as a family report rather than one checkpoint: very large mostly Chinese/English pretraining, then multi-stage human-feedback alignment, then a tool-using "All Tools" branch and smaller open 9B variants.
- **Guideline:** When a lab reports a whole family under one name, separate the family-level training worldview from the exact open checkpoint you can actually inspect.
- **Authors / Lab:** Team GLM / Zhipu AI
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2406.12793
- **Relevant topics:** Chinese-English frontier models, family-level model report, multi-stage alignment, tool use, long context

## Abstract
The public `GLM-4` report is a family report covering `GLM-4`, `GLM-4-Air`, `GLM-4-9B`, and `GLM-4 All Tools`. It describes a training worldview built from earlier ChatGLM generations: pretraining on **ten trillions of tokens**, mostly Chinese and English with smaller amounts from 24 additional languages, followed by **multi-stage post-training** with supervised finetuning and learning from human feedback. The open checkpoint most users can directly inspect is `GLM-4-9B-Chat`, but the report's main value is the broader family-level recipe.

## Key Contributions
- Family-wide pretraining on **ten trillions of tokens**.
- Explicitly **Chinese-English-first** alignment, with some multilingual extension.
- Multi-stage post-training using **SFT + human feedback**.
- A distinct **GLM-4 All Tools** line aligned to choose and use tools autonomously.
- Strong public emphasis on **long context**, **function calling**, and **tool use** in the 9B open line.

## Key Figures/Tables to Study
- The abstract is again unusually dense and useful: it states token scale, language mix, and post-training philosophy.
- The `GLM-4-9B-Chat` model card is the clearest source for the open checkpoint's capabilities: **128K context**, **26 languages**, **tool calling**, and **1M-context** derivative release.

## Technical Details

### Naming / scope note
- `GLM-4` is **ambiguous by design** in the public record: sometimes it means the **family**, sometimes a larger closed model, and sometimes people really mean the open **GLM-4-9B** line.
- This page follows the checklist and uses `GLM-4` as the **family-level report**, while pulling concrete open-checkpoint facts from **GLM-4-9B-Chat**.

### Pretraining
- The report says GLM-4 models were pre-trained on **ten trillions of tokens**.
- Data is **mostly Chinese and English**, plus a smaller corpus from **24 additional languages**.
- The family report presents GLM-4 as the cumulative result of lessons from previous **ChatGLM** generations.

### Post-training
- Publicly described as a **multi-stage post-training process** involving **supervised fine-tuning** and **learning from human feedback**.
- The report does not publicly expose the same level of optimizer / hyperparameter detail as fully open post-training reports like Tulu 3.
- `GLM-4 All Tools` is aligned specifically to infer **when** and **which** tools to call, including web browsing, Python, image generation, and user-defined functions.

### Open 9B line
- The `GLM-4-9B-Chat` card describes the open model as the latest generation of the GLM-4 series.
- Public capabilities include **128K context**, **web browsing**, **code execution**, **function calling**, and **26-language** support.
- Zhipu also released a **1M-context** variant and a multimodal **GLM-4V-9B** branch.

### Why it matters
- GLM-4 is useful because it is one of the clearest public Chinese-English family reports that connects **foundation pretraining scale** to **tool-using aligned agents**.
- It is less reproducible than Allen AI-style releases, but stronger than a pure product blog because it still provides a real family report.

## Connections
- [[qwen-2.5]] is the closest large Chinese/English open-family comparison.
- [[yi]] is an earlier data-centric bilingual family with less emphasis on tool use.
- [[bfcl]] and [[toolformer]] matter because GLM-4's open 9B line explicitly foregrounds function calling and tool selection.
