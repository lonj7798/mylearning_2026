<!-- scope: Magicoder / OSS-Instruct — seed from real open-source snippets to synthesize coding instructions
     deps: [[self-instruct]], [[evol-instruct]]
     see-also: [[wizardcoder]], [[code-evol-instruct]], [[opc-synthetic-code]]
-->

# Magicoder / OSS-Instruct: Empowering Code Generation with Open-Source Code Seeds
- **Core Insight:** Seeding synthesis with *real* open-source code snippets forces the teacher LLM off its mean-response attractor — the model is compelled to ground the new problem in the seed's idiosyncrasies, yielding lower-bias and more diverse coding instructions than pure self-instruct.
- **Guideline:** To generate coding SFT data, sample small random open-source snippets (The Stack, StarCoderData), pair each with a prompt that asks the teacher to invent a realistic programming problem inspired by the snippet; yields 75K–100K instructions sufficient to close the gap to ChatGPT on HumanEval+.
- **Authors:** Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang (UIUC)
- **Year:** 2023/2024 (ICML 2024)
- **URL:** https://arxiv.org/abs/2312.02120
- **Relevant topics:** synthetic code instructions, OSS-Instruct, Magicoder

## Abstract
OSS-Instruct generates instruction-tuning data for code by giving a teacher LLM a random 1–15-line open-source code snippet and asking it to construct a new, self-contained programming problem inspired by the snippet. Because the seed is real-world code, the resulting problems inherit realistic structure, API usage, and bug patterns — sidestepping the mean-regression of Self-Instruct and the teacher-preference drift of Evol-Instruct. Magicoder-CL-7B and MagicoderS-CL-7B (CodeLlama-7B base, fine-tuned on 75K OSS-Instruct data + additional Evol-Instruct) close the gap to top proprietary code models and surpass ChatGPT on HumanEval+ (66.5 vs 65.9 pass@1) while being fully open-source (code + weights + data).

## Key Contributions
- Introduced **OSS-Instruct**: real snippet → teacher-generated instruction, orthogonal to Self/Evol-Instruct.
- Open-released **Magicoder** series (7B code LLMs) with competitive HumanEval/HumanEval+/MBPP numbers.
- Showed OSS-Instruct + Evol-Instruct combine additively.
- Demonstrated the teacher's "grounding" in a random snippet measurably broadens the instruction distribution (category entropy, lexical diversity).

## Key Figures/Tables to Study
- **Figure showing a snippet → instruction example** — two unrelated code fragments combined into a plausible ML problem.
- **Table of HumanEval / HumanEval+ / MBPP / MBPP+ / DS-1000** comparing Magicoder vs WizardCoder vs code-davinci-002 vs ChatGPT.
- **Ablation: OSS-Instruct only vs + Evol-Instruct** — both matter, combination wins.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:** random 1–15-line snippets drawn from StarCoderData (filtered subset of The Stack). Snippets span many languages but the pipeline emphasizes Python.

- **Generation step(s):**
  - Prompt template (sketch): *"Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present the problem in the following format: [Problem Description] ... [Solution] ..."*  followed by the snippet.
  - Teacher (GPT-3.5-Turbo in the released dataset) emits a problem description plus a reference solution.
  - The output pair is post-processed into a Magicoder-style instruction (`Question: ...`, `Answer: ...`).

- **Filtering/rescoring:** decontamination against HumanEval / MBPP (n-gram overlap); dedup; language/format validation.

- **Output shape:** 75K OSS-Instruct instruction-solution pairs released publicly under MIT license. MagicoderS uses an additional ~110K from [[evol-instruct]]-style code evolution on top of OSS-Instruct.

- **Teacher model(s):** GPT-3.5-Turbo for the released dataset; method applicable to any capable code LLM.

- **Cost estimate:** disclosed as a few hundred USD of API for 75K examples — one of the cheapest notable synthetic datasets.

## Quality / diversity evaluation
- MagicoderS-CL-7B: HumanEval+ 66.5 pass@1 (vs ChatGPT 65.9, vs WizardCoder-CL-7B 51.8).
- MBPP+ 49.3 (SOTA among open 7B at release).
- Lexical diversity of OSS-Instruct instructions >> Self-Instruct / Evol-Instruct on the same base teacher — explicitly measured.
- Category distribution (string ops, math, data structures, …) more balanced than Code-Evol-Instruct.

## Risks + gotchas
- **Snippet license leakage:** open-source snippets can be GPL; outputs inspired by GPL code have been legally argued both ways.
- **Teacher anchoring is imperfect:** the teacher sometimes ignores the snippet and generates a default-style problem.
- **Language coverage:** released data is Python-heavy; multilingual code requires re-running the pipeline per language.
- **Benchmark contamination risk** as with all synthetic SFT — the paper documents decontamination steps but later studies found residual overlap.

## Connections
- Third pole of code synthesis alongside [[wizardcoder]] (Evol-Instruct operators on code) and seed-based [[self-instruct]].
- Dataset used in many open models; precursor to [[opc-synthetic-code]].
- Combinable with [[evol-instruct]] (Magicoder-S uses both).
- Diversity claim ties to [[prismatic-synthesis]]'s gradient-diversity framing — OSS-Instruct empirically produces broader gradient coverage than Self-Instruct.
