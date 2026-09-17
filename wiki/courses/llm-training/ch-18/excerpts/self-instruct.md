---
chapter: ch-18
course: llm-training
phase: read
excerpt_of: "Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang, Kordi, Mishra, Liu, Smith, Khashabi, Hajishirzi)"
source_url: https://arxiv.org/abs/2212.10560
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Self-Instruct — the steps and numbers ch-18 uses

Rewritten for the 2026-09 revision of ch-18. Every number was read in the arXiv:2212.10560 PDF (ACL 2023 version) on
2026-09-15. The earlier version of this excerpt contained unsupported numbers (a "~252K raw" count, a "~50%" ROUGE-L
rejection share, a root-verb-entropy ablation, and the text-davinci-001 generator); see ch-18 read.md Corrections 1-5.

- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Year:** 2022 (arXiv v1 2022-12; ACL 2023)
- **Source type:** paper

## Pipeline (§2.2)
1. Task pool starts with 175 human-written tasks (1 instruction and 1 instance each), written by the authors without
   reference to the test sets (footnote 3).
2. Instruction generation: 8 instructions sampled from the pool as in-context examples, 6 human-written and 2 generated.
3. Classification identification, then instance generation (output-first for classification tasks, input-first otherwise).
4. Filtering: a new instruction enters the pool only if its ROUGE-L similarity with every existing instruction is below
   0.7; instructions with keywords such as "image, picture, graph" are excluded; duplicate instances and instances with
   the same input but different outputs are removed; heuristics remove too-long or too-short instructions and outputs
   that repeat the input.

## Generator and data (§3, Table 1)
- Generator: the largest GPT-3 model, "davinci" engine, via the OpenAI API. Comparisons are with text-davinci-001
  (InstructGPT-001) unless stated (footnote 1). Generating the data set cost about $600 (App. A.2).
- Output: 52,445 instructions (11,584 classification), 82,439 instances (35,878 with empty input).
- Diversity (§3.2): 26,559 of 52,445 instructions have a parsable verb-noun structure; Fig. 4 shows the ROUGE-L
  overlap of generated instructions with their most similar seed.
- Quality (§3.3, Table 2), 200 sampled instructions with 1 instance each: valid task 92%; appropriate input 79%;
  correct output 58%; all fields valid 54%.

## Training and results
- GPT3-Self-Inst: GPT-3 davinci fine-tuned through the OpenAI API for 2 epochs with prompt loss weight 0 (§4.1).
- Unseen SuperNI tasks (§4.3, Table 3; 119 tasks, ROUGE-L): GPT-3 6.8; GPT3-Self-Inst 39.9; InstructGPT-001 40.8.
- 252 expert-written user-oriented instructions, human rating (§4.4): 5% absolute gap behind InstructGPT-001 (Abstract).
- Data size and quality (§4.5, Fig. 7): top-rated responses rise from 31.0% (175 seed tasks only) to 44.4%
  (51,200 instructions), almost plateauing after 16K; regenerating outputs with InstructGPT-003 gives 54.4%.

## Limits relevant to ch-18
- No correctness check beyond heuristics; the quality audit is on 200 samples by one expert annotator (an author).
- Tülu 1 later found that SFT on this data set lowered LLaMA 13B MMLU from 42.3 to 30.4 ([[tulu-1-how-far-can-camels-go]] Table 3).
