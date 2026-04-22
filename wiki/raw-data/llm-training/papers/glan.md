<!-- scope: taxonomy-driven synthetic instruction tuning data, no seed examples
     deps: [[self-instruct]]
     see-also: [[persona-hub]], [[phi-1-5]], [[evol-instruct]]
-->

# GLAN: Generalized Instruction Tuning via Taxonomy-Driven Synthesis
- **Core Insight:** You don't need seed examples *or* real user data to produce broad-coverage instruction tuning data — a single pre-curated taxonomy of human knowledge (fields → subfields → disciplines → subjects → syllabus → concepts) gives the teacher LLM enough structure to emit diverse, discipline-balanced instructions from scratch.
- **Guideline:** When instruction data for a target capability is scarce or biased, (1) build/borrow a multi-level discipline taxonomy, (2) expand each leaf into a syllabus of class sessions with explicit concept lists, (3) prompt a teacher to generate instructions keyed to each concept; the breadth is given to you by the taxonomy, not the seed pool.
- **Authors:** Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, Furu Wei (Microsoft Research + collaborators)
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2402.13064
- **Relevant topics:** synthetic instruction data, taxonomy-driven synthesis, no-seed instruction tuning

## Abstract
Prior instruction-tuning data synthesis (Self-Instruct, Evol-Instruct) is seeded by a small real instruction pool and inherits its biases. GLAN replaces the seed with a **taxonomy**: human knowledge is decomposed semi-automatically into fields → subfields → disciplines. Each discipline gets an auto-generated subject list; each subject gets an auto-generated syllabus of class sessions; each class session is enumerated as a concept list. Instructions are generated at the concept level, guaranteeing coverage across all branches. A Mistral-7B fine-tuned on the resulting dataset excels in math reasoning, coding, logical reasoning, academic exams, and general instruction following — without any task-specific real data. Adding a new capability = adding a taxonomy node.

## Key Contributions
- Established **"taxonomy-as-seed"** as a third synthesis paradigm alongside Self-Instruct (seed-plus-bootstrap) and Magpie (no-seed-at-all).
- Four-level hierarchy: Field → Subfield → Discipline → Subject → Session → Concept → Instruction.
- Demonstrated fine-grained coverage control: adding a taxonomy node adds a capability.
- Showed strong results on math, code, logic, MMLU-style exams from a single unified synthesis pass.

## Key Figures/Tables to Study
- **Figure 1 / taxonomy diagram** — the four-level hierarchy and its expansion ratios.
- **Table comparing GLAN to Alpaca / Evol-Instruct / WizardLM data** on downstream Mistral-7B.
- **Category-coverage bar chart** — GLAN's discipline coverage vs seed-based baselines.

## Synthesis pipeline (REQUIRED — be concrete)
- **Seed input:**
  - A top-level field list (e.g. Mathematics, Computer Science, Humanities, Natural Sciences, …) — a few dozen entries seeded by the authors + GPT-4.
  - No real instruction seeds; no user logs; no existing dataset.

- **Generation step(s):**
  1. **Field → Subfield → Discipline:** prompt GPT-4 to decompose each field into subfields and then disciplines (taxonomy construction — semi-automatic, curated).
  2. **Discipline → Subjects:** for each discipline, prompt for a comprehensive list of subjects.
  3. **Subject → Syllabus:** prompt for a multi-class-session syllabus with explicit learning objectives.
  4. **Session → Concepts:** prompt for the key concepts covered in each session.
  5. **Concept → Instructions:** for each concept, prompt for instruction-response pairs at varied difficulty levels; include verification-friendly answer formats for math/code.
  - Each step uses GPT-4 (or GPT-4-Turbo) with templated prompts.

- **Filtering/rescoring:** deduplication + light decontamination; some quality filtering via teacher self-verification for math/code. Heavier filter methods are explicitly *not* required by GLAN's design because the taxonomy already controls coverage.

- **Output shape:** multi-million instruction-response pairs spanning all leaf concepts; released alongside the paper for research use.

- **Teacher model(s):** GPT-4 / GPT-4-Turbo (taxonomy curation + instruction synthesis).

- **Cost estimate:** not disclosed per step; GPT-4 API cost is the dominant line item.

## Quality / diversity evaluation
- Mistral-7B + GLAN outperforms same-base models fine-tuned on Alpaca / WizardLM / CodeAlpaca on MATH, GSM8K, HumanEval, MBPP, BBH, ARC, MMLU.
- No task-specific data used — generalization attributed to coverage.
- Ablations on taxonomy depth confirm deeper trees give flatter capability distributions.

## Risks + gotchas
- **Teacher-bias concentrates at the top:** GPT-4's view of what constitutes a field/discipline shapes the entire downstream corpus.
- **Taxonomy audit cost:** discipline-level coverage is only as balanced as the curator's review.
- **Limited novel-capability generation:** GLAN does not invent new skills the teacher lacks; like [[magpie]], it extracts what's already in the teacher.
- **License encumbrance** through GPT-4 outputs.

## Connections
- Third pole of the synthesis space: seed-based ([[self-instruct]], [[evol-instruct]]) vs no-seed ([[magpie]]) vs taxonomy-driven (GLAN).
- Complements [[persona-hub]]: persona diversifies "who asks," GLAN diversifies "what is asked."
- Precursor of open reproductions like [[hf-cosmopedia]]'s taxonomy layer.
- Feeds the "taxonomy + gradient-space coverage" research direction (compare to [[prismatic-synthesis]]).
