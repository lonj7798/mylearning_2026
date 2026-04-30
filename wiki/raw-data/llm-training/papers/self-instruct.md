<!-- scope: bootstrap-from-LM-outputs pipeline to generate instruction-tuning data
     deps: [[README]]
     see-also: [[alpaca]], [[evol-instruct]], [[humpback]]
-->

# Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Core Insight:** A large LM can bootstrap its own instruction-tuning dataset from ~175 seed tasks, reaching InstructGPT-001 quality without any private human annotation.
- **Guideline:** If you need instruction data, start with a small diverse seed set, generate candidate instructions + inputs + outputs with a strong LM, and filter by ROUGE-L overlap and format validity — aim for tens of thousands of accepted instances.
- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi (Yejin Choi group / UW / AI2)
- **Year:** 2022
- **URL:** https://arxiv.org/abs/2212.10560
- **Relevant topics:** SFT data construction, synthetic data, bootstrapping, instruction tuning

## Abstract
Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning.

## Key Contributions
- A **four-step pipeline** (seed → instruction gen → instance gen → filter) that produces instruction data from a single LM.
- Demonstrated **+33 absolute points on Super-NaturalInstructions** when applied to vanilla GPT-3, matching InstructGPT-001 (trained with private human data).
- Released the synthetic dataset — the ancestor of Alpaca, Vicuna, and every subsequent "X-Instruct" derivative.
- Established the template (seed tasks + ROUGE-L dedup + format filters) still used in 2024-era data construction.

## Key Figures/Tables to Study
- **Figure 1** — the four-stage pipeline diagram (instruction generation, classification/non-classification branch, instance generation, filter).
- **Table 1 / Table 2** — task-set diversity stats before vs after bootstrapping.
- **Performance table on Super-NI** — Self-Instruct-tuned GPT-3 vs InstructGPT-001 vs vanilla GPT-3.

## Technical Details
**Seed pool:** 175 human-written tasks (1 instruction + 1 instance each) covering classification, generation, open-ended, extraction.

**Pipeline:**
1. **Instruction generation** — prompt the LM with 8 in-context examples (6 from seed, 2 from previously generated) and ask for a new task instruction.
2. **Classification-vs-non-classification branching** — ask the LM whether the instruction is a classification task; this changes the instance-generation prompt template (input-first for classification to avoid label bias, output-first otherwise).
3. **Instance generation** — for each accepted instruction, prompt the LM to produce an input and an output.
4. **Filtering**:
   - Drop instructions with **ROUGE-L > 0.7** to any existing instruction (diversity filter).
   - Drop instances where input == output, outputs too long/short, or the instruction contains "image/graph/file".
   - Drop ill-formatted generations.

**Final dataset:** ~52K instructions × ~82K instances (after filtering from ~252K raw generations) produced using GPT-3 (text-davinci-001-era model).

**Prompt template sketch (instruction generation):**
```
Come up with a series of tasks:
Task 1: <seed 1>
Task 2: <seed 2>
...
Task 8: <seed 8>
Task 9:
```

**Failure modes observed:** hallucinated "impossible" tasks, output-bias in classification, repetition — addressed via the diversity filter.

## Connections
- Direct ancestor of [[alpaca]] (same method, run on text-davinci-003), [[evol-instruct]] (adds complexity dimension), and [[humpback]] (reverses the direction: text → instruction).
- Conceptual cousin of [[star]]: both bootstrap from model-generated intermediate traces.
- The 175-seed + ROUGE-filter template shows up in virtually every open SFT pipeline — see `[[tulu-3]]`.
