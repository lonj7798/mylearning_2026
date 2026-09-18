<!-- scope: Code Evol-Instruct, the code-instruction evolution method of the WizardCoder paper (§3.1, §3.2, §5, App. C–E): prompt template, five heuristics, round/stop rule, decontamination, round and evolver ablations. Model results are in [[wizardcoder]].
     deps: [[evol-instruct]], [[self-instruct]]
     see-also: [[wizardcoder]], [[oss-instruct]], [[opc-synthetic-code]], [[nemotron-4-synthetic]]
-->

# WizardCoder: Empowering Code Large Language Models with Evol-Instruct
- **Core Insight:** When each evolution round's data is used alone at a matched size, fine-tuning on one evolved round of 18.8k–19.7k samples gives HumanEval pass@1 of 51.2–56.1, against 45.7 for the 20.0k Code Alpaca seed data (§5, Table 5, sample-matched rows).
- **Guideline:** When evolving code instructions with an LLM, apply the five code heuristics round by round, merge all rounds with the seed data, and stop when pass@1 on a held-out dev set drops, because in the paper's round ablation pass@1 on the MBPP-400 dev set and on HumanEval peaked after three rounds (§3.2, §5, Fig. 3); v2 uses an external dev set for this stop rule, whereas v1 used HumanEval pass@1, a reported test benchmark (v1 §3.2).
- **Authors:** Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, et al. (10 authors; Microsoft, Hong Kong Baptist University)
- **Year:** 2023 (arXiv v1 2023-06; v2 2025-05; ICLR 2024)
- **URL:** https://arxiv.org/abs/2306.08568 ; repository named in v1: https://github.com/nlpxucan/WizardLM
- **Source type:** paper
- **Relevant topics:** Code Evol-Instruct, code instruction synthesis, instruction complexity, evolution stop rule, decontamination, code SFT data

## Abstract
Instruction fine-tuning is less studied for code LLMs than for general LLMs. The paper presents Code Evol-Instruct, which adapts Evol-Instruct (LLM rewriting of instructions into harder versions) to code, and uses the evolved data to fine-tune StarCoder and CodeLlama into WizardCoder models. On HumanEval, HumanEval+, MBPP, DS-1000, and MultiPL-E the models outperform other open-source code LLMs. WizardCoder 15B exceeds Claude and Bard on HumanEval and HumanEval+, and WizardCoder 34B is comparable to GPT-3.5 on HumanEval and exceeds it on HumanEval+. A preliminary analysis points to instruction complexity as the main factor.

## Key Contributions
- A single evolution prompt template with five code-specific heuristics, including erroneous code as an adversarial reference and time/space complexity requirements (§3.1).
- An iterative procedure: evolve, merge all rounds with the seed data, fine-tune, and stop when dev-set performance drops (§3.2).
- Ablations separating complexity from quantity (Table 5) and from test-set similarity (Fig. 4), and a comparison of evolver models (Table 4).
- A decontamination step using embedding retrieval plus a GPT-4 match decision (App. C).

## Key Figures/Tables to Study
- **§3.1 prompt boxes:** the evolution template and the five heuristics, word for word.
- **Figure 3:** pass@1 against number of evolution rounds (MBPP-400 dev, HumanEval).
- **Table 4:** evolver model (GPT-4, GPT-3.5, CodeLlama) × base model. **Table 5:** per-round data at matched samples and tokens.
- **Figure 4 / App. C:** GPT-4 similarity (1–10) between HumanEval items and top-1 retrieved training items per round. **App. D:** three seeds evolved over rounds 0–3.

## Technical Details
- **Evolution prompt (§3.1).** "Please increase the difficulty of the given programming test question a bit. You can increase the difficulty using, but not limited to, the following methods: {method} {question}". {question} = the current instruction; {method} = one heuristic.
- **The five heuristics (§3.1), as printed.** (1) "Add new constraints and requirements to the original problem, adding approximately 10 additional words." (2) "Replace a commonly used requirement in the programming task with a less common and more specific one." (3) "If the original problem can be solved with only a few logical steps, please add more reasoning steps." (4) "Provide a piece of erroneous code as a reference to increase misdirection." (5) "Propose higher time or space complexity requirements, but please refrain from doing so frequently."
- **Differences from general Evol-Instruct (v1 §3.1).** Removed the deepening, complicating-input, and In-Breadth Evolving instructions; unified the prompt template; added code debugging and time-space complexity constraints.
- **Heuristic selection per question.** Neither v1 nor v2 states how {method} is chosen (random or otherwise).
- **Seed data.** Code Alpaca, about 20k samples (v2 §3.2; "20,000 samples" in v1 §3.2), which was generated with self-instruct (v1 §3).
- **Merging.** After each round, the evolved data from all previous rounds is merged with the original data for fine-tuning (§3.2).
- **Cumulative sizes.** 38k after round 1, 58k after round 2, 78k after round 3, 98k after round 4; the round-3 data (about 78k) is the final set (v1 §4.5; v2 §4.2).
- **Stop rule.** v2: "An external dev set serves as the controlled Evol Stop. If the performance drops, we halt the evolution" (§3.2); the round ablation uses MBPP train + dev merged into "MBPP-400" (§5). v1: stop when HumanEval pass@1 declines and keep the model with the highest HumanEval pass@1 (v1 §3.2).
- **Evolver and responder.** OpenAI gpt3.5-turbo evolves instructions and writes responses (v2 §4.2).
- **Evolver ablation (Table 4, HumanEval pass@1).** StarCoder-15B base: GPT-4 evolver 62.2, GPT-3.5 59.8, CodeLlama 55.5. CodeLlama-34B base: 73.8, 73.2, 70.1. The authors note GPT-4's larger coding advantage (88.4 vs 73.2) does not carry over proportionally (§5). For the open-source evolver, CodeLlama-Instruct-34B was first fine-tuned on Code Alpaca to write responses (App. E).
- **Complexity vs quantity (Table 5, HumanEval pass@1; each round trained alone; base model not stated).** Matched samples: round 0 20.0k → 45.7; round 1 18.8k → 56.1; round 2 19.7k → 53.0; round 3 19.3k → 54.3; round 4 19.0k → 51.2. Matched 2.3M tokens: 44.5, 51.8, 52.4, 50.0, 49.4. Combining rounds gives the best result (§5).
- **Decontamination (App. C).** gte-large embeddings retrieve the top 5 training samples for each test sample; GPT-4 makes a yes/no match decision; matched samples are removed.
- **Similarity check (§5, Fig. 4, App. C).** GPT-4 scores the similarity of each HumanEval item and its top-1 retrieved training item from 1 (different) to 10 (identical); evolution does not increase the score, and scores stay low in all rounds.
- **Fine-tuning format (§3.2).** Alpaca-style prompt: "Below is an instruction that describes a task ... ### Instruction: {instruction} ### Response:".

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WizardCoder 15B (StarCoder 15B base); WizardCoder 34B (CodeLlama-34B-Python base) | 15B; 34B | SFT | training data | Code Evol-Instruct data, about 78k samples | arXiv:2306.08568v2 §4.2 | verified 2026-09-14 | Fig. 3 (v2) / v1 §4.5: best pass@1 after 3 rounds |
| same | 15B; 34B | SFT | seed data | Code Alpaca, about 20k samples | v2 §3.2 | verified 2026-09-14 | Table 5: seed round 45.7 vs evolved rounds 51.2–56.1 |
| same | 15B; 34B | SFT | evolver / response model | gpt3.5-turbo | v2 §4.2 | verified 2026-09-14 | Table 4: GPT-3.5 evolver 59.8 (15B) / 73.2 (34B); GPT-4 evolver 62.2 / 73.8 |
| same | 15B; 34B | SFT | batch size (unit not stated); sequence length | 512; 2048 | v2 §4.2; v1 §4.2 (15B) | verified 2026-09-14 | no ablation reported |
| same | 15B; 34B | SFT | steps; warmup; peak LR; schedule; precision | 200; 30 steps; 2e-5; cosine; fp16 | v2 §4.2; v1 §4.2 (15B) | verified 2026-09-14 | no ablation reported |
| same | 15B; 34B | SFT | epochs, optimizer, weight decay, loss masking, packing | not reported | checked v1 and v2 body and appendices | not reported | none |

## Findings relevant to generality
- **Held-out suites.** Training data is evolved from Code Alpaca, and evaluation covers HumanEval, HumanEval+, MBPP, DS-1000 (7 libraries), and MultiPL-E (8 languages) (§4.3–§4.5). WizardCoder 34B is above CodeLlama-Instruct-34B in all 8 MultiPL-E languages (Table 2); WizardCoder 15B DS-1000 insertion score is 32.8 vs StarCoder 25.4 (Table 3).
- **Headline scores (Table 1, n = 200, temperature 0.2, top-p 0.95).** WizardCoder 15B HumanEval 57.3, MBPP 51.8; WizardCoder 34B 71.5, 61.2. Greedy decoding, EvalPlus leaderboard setting: 34B HumanEval+ 64.6 vs GPT-3.5 63.4; 15B 59.8 vs Claude-Plus 53.0 and Bard 44.5, where the sentence does not name which of HumanEval or HumanEval+ is meant (§4.3, Fig. 2).
- **Selection on a test benchmark (v1).** v1 chose the number of rounds and the final model by HumanEval pass@1, which is also a reported benchmark (v1 §3.2, §4.5). v2 describes an external dev set and MBPP-400 instead (§3.2, §5).
- **Not reported:** effect on non-code tasks, forgetting, output diversity, per-heuristic ablation.

## Connections
- [[wizardcoder]] — card for the same paper's model results; this card covers the data method.
- [[evol-instruct]] — the general-domain method adapted here; v1 §3.1 lists the removed operators.
- [[self-instruct]] — method that produced the Code Alpaca seed data (v1 §3).
- [[oss-instruct]] — Magicoder generates code instructions from open-source snippets and further fine-tunes on evol-codealpaca-v1, an open-source Evol-Instruct dataset of about 110K samples (arXiv:2312.02120v2 §1, §3).
- [[opc-synthetic-code]] — OpenCoder's stage-2 instruction data includes a 111K Evol-Instruct subset that cites this paper (arXiv:2411.04905v3 §4.1, Table 5).
- [[nemotron-4-synthetic]] — Nemotron-4 340B code SFT uses Genetic Instruct, which combines self-instruction with "wizard coder mutations" (arXiv:2406.11704v2 §3.3.1).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2306.08568 (arXiv v2, 2025-05-27, ICLR 2024 version; and v1, 2023-06-14).
- Corrections to the previous card version:
  - Title "Code Evol-Instruct: Code-Specialized Evolution Operators" (not a published title) → exact paper title; scope line records that the card covers the method sections.
  - Operator catalog of 5 "axes" with 14 operators → the paper lists exactly five heuristics (§3.1).
  - "randomly pick one operator" → selection rule not stated in v1 or v2.
  - "M = 3 iterations per seed" → whole-dataset rounds with merging; 3 rounds chosen by ablation (§3.2, §5; v1 §4.5).
  - "Teacher GPT-3.5/4" → gpt3.5-turbo for released data; GPT-4 only in the Table 4 ablation (§4.2, §5).
  - "benchmark-overlap pruning by HumanEval/MBPP n-gram match" → gte-large top-5 retrieval plus GPT-4 yes/no match (App. C).
  - URL "github.com/nlpxucan/evol-instruct" → repository named in v1 is github.com/nlpxucan/WizardLM.
- Removed as unsupported by the source: "about 4× expansion"; pruning of identical, refused, and length-degenerate evolutions; format normalization and code-block extraction steps; "domain axis" operators (data science, systems, multi-language ports); "erroneous code highlighted as particularly valuable"; Genetic Instruct and Evolutionary Contrastive Distillation descriptions; smol-talk link via Self-OSS-Starcoder2-Instruct; risks on operator bias, GPT-4 refusal drift, and library-driven contamination.
