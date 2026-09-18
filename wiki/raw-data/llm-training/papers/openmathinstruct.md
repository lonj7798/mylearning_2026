<!-- scope: OpenMathInstruct-1 (NVIDIA, Feb 2024) — 1.8M code-interpreter math solutions sampled from Mixtral-base on GSM8K/MATH train problems, masked-text prompting, fair downsampling, and the OpenMath SFT models
     deps: [[self-instruct]]
     see-also: [[openmathinstruct-2]], [[mammoth]], [[metamath]], [[numina-math]], [[training-verifiers-to-solve-math-word-problems]]
-->

# OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset
- **Core Insight:** Sampling up to 896 (MATH) and 256 (GSM8K) code-interpreter solutions per training problem from the permissively licensed Mixtral-base model, with masked reference solutions in the prompt, covers 93.0% of MATH and 99.9% of GSM8K training problems; CodeLlama-70B fine-tuned on a 1.02M subset reaches 84.6% GSM8K and 50.7% MATH with greedy decoding (§1, Table 2, Table 3).
- **Guideline:** When an open generator solves fewer training problems than a closed model, put the reference text solution in the few-shot prompt with its intermediate numbers masked, because masking raised MATH coverage from 80.1% to 85.9% (Table 2) and the fine-tuned Mistral-7B from 36.9 to 37.4 on the MATH validation subset (Table 5), while unmasked references produced answer-copying shortcut solutions (§2.2.3).
- **Authors:** Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, Igor Gitman (NVIDIA)
- **Year:** 2024 (arXiv v1 2024-02-15; v2 2024-11-03 lists NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2402.10176 ; dataset https://huggingface.co/datasets/nvidia/OpenMathInstruct-1
- **Source type:** paper (with dataset card)
- **Relevant topics:** math reasoning, synthetic solution generation, code-interpreter (tool-integrated) solutions, training-set coverage, data selection, permissive licensing

## Abstract
Existing large math instruction-tuning datasets such as MetaMathQA and MAmmoTH are built from closed models with commercially restrictive licenses. The authors attribute the lack of open generators to the gap in math skill between GPT-4 and open models. Using a new prompting method, recent open models, and brute-force scaling, they build OpenMathInstruct-1: 1.8M problem–solution pairs in code-interpreter format for GSM8K and MATH, generated with Mixtral. OpenMath-CodeLlama-70B, trained on a subset, scores 84.6% on GSM8K and 50.7% on MATH, which the authors describe as competitive with the best GPT-distilled models. Code, models, and data are released under a commercially permissive license.

## Key Contributions
- A 1.8M-pair math SFT dataset from an open generator, at least four times larger than prior math fine-tuning sets (§1, Table 1), plus 6.6M released incorrect solutions (§1, footnote 4).
- Masked text solution prompting (Mask-Text): reference solutions with intermediate computations replaced by symbols (§2.2.3, App. B.5).
- Fair (round-robin per problem) downsampling and code-preferential selection (Any-Code, Majority-Code) (§2.4).
- OpenMath models from 7B to 70B on Mistral, Llama 2, and CodeLlama bases, with ablations on selection, prompting, and data size (§4, Tables 3–8).

## Key Figures/Tables to Study
- Figure 1 (coverage vs samples, Mixtral vs GPT-4); Table 2 (samples, unique solutions, coverage per prompt); Table 3 (main results on 7 benchmarks); Tables 4–8 (ablations); Tables 9–10 (error analysis); Table 12 (training hyperparameters).

## Technical Details
**Generation (§2.1)**
- Training set coverage (TSC) is the share of training problems with at least one generated solution that reaches the ground-truth answer (§1).
- Generator: Mixtral-base (§1, §2.1); the dataset card links Mixtral-8x7B-v0.1. Mixtral needs almost 8× GPT-4's samples to match GPT-4's GSM8K coverage and stays below GPT-4 on MATH at 12× (§1, Fig. 1).
- Prompt: instruction I plus K = 5 training-set examples in code-interpreter format, then the target question; the answer goes in `\boxed{}` (§2.1, Table 13). A generated solution is kept if it reaches the correct answer (§2.1).
- Format: text interleaved with `<llm-code>` blocks; the executed output is inserted between `<llm-code-output>` tags and generation resumes (§2.1). The code output is not masked during training (§2.1, footnote 6).
- Sampling: temperature 1.0, top_p 0.95; 4,096 total input-output tokens; at most 512 new tokens after each code block; at most 3 code blocks; generation stops after any code execution error; TensorRT-LLM (§2.1). Code blocks time out after 10 seconds (dataset card).

**Prompting strategies (§2.2, Table 2)**
- Default: MATH 224 samples per problem, 177K unique correct solutions, 80.1% coverage; GSM8K 128 samples, 434K, 99.1%.
- Subject-specific (Subj, MATH only): 7 subject prompts with one example per difficulty level, 32 samples each (224 total); 191K solutions, coverage 80.1%. Combined with Default: 85.1% coverage at 448 samples per problem (§2.2.2).
- Mask-Text: MATH 224 samples, 192K solutions, 85.9%; with Subj 227K, 87.5%; GSM8K 128 samples, 602K, 99.9%. Masked solutions are generated few-shot: 8 candidates, drop length outliers and candidates containing the final answer, rank by fewest numbers (App. B.5).
- Totals: MATH 896 samples, 787K unique solutions, 6,978 of 7,500 problems; GSM8K 256 samples, 1.04M solutions, 7,469 of 7,473 problems (§2.2.3, Table 2).

**Post-processing and statistics**
- Remove solutions with multiple `\boxed{}` blocks or an unclosed `<llm-code>`; trim text after the answer line (§2.3). Correct-answer solutions with flawed reasoning are not filtered; the authors find them "rare" anecdotally (§2.3).
- Code blocks per solution: 0 in 16.4%, 1 in 81.7%, 2 or more in the remaining 2% (App. A.1). Text-only solutions: 2% of GSM8K, 35.1% of MATH (§2.4.2).
- Solution counts: 57.4% of GSM8K problems (4,292 of 7,473) have more than 128 valid solutions; 19% of covered MATH problems (1,324 of 6,978) have ≤ 10 valid solutions out of 896 samples (App. A.2).

**Ablations (Mistral-7B, 1K validation subsets from the training sets; §4.1)**
- Fair vs naive downsampling, 128K: GSM8K 75.3 vs 74.3, MATH 37.0 vs 35.0 (Table 4).
- Masked vs default prompting, 128K: GSM8K 77.7 vs 73.8, MATH 37.4 vs 36.9 (Table 5).
- Data size 128K / 256K / 512K: GSM8K 75.3 / 79.0 / 81.0, MATH 37.0 / 38.6 / 41.6; more training steps gave no benefit (§4.1.3, Table 6).
- MATH-only, 128K: Subject vs Default prompts, pass@1 38.3 vs 39.1, SC (k=4) 44.5 vs 41.7 (Table 7); Default / Majority-Code / Any-Code, pass@1 37.4 / 39.8 / 39.4, SC (k=4) 45.2 / 42.6 / 42.6 (Table 8). Any-Code was chosen because it is smaller (512K vs 664K) (§4.1.4).

**Main results (§4, Table 3; greedy unless noted)**
- OpenMath-Mistral-7B: GSM8K 80.2, MATH 44.5, GSM-Hard 63.7; SC (k=50) 86.9 / 57.2.
- OpenMath-CodeLlama-70B: GSM8K 84.6, MATH 50.7; SC (k=50) 90.8 / 60.4. OpenMath-Llama2-70B: 84.7 / 46.3. ToRA-70B (Llama-2): 84.3 / 49.7. GPT-4 Code Interpreter: 97.0 / 69.7.
- Analysis of the 512K model on MATH validation: Level 1 72.4%, Level 5 16.3%; geometry is the weakest subject (§5, Fig. 4). Code + text solutions 45.3% (722) vs text-only 32.0% (278) (Table 9). Of 584 errors, 292 are code reasoning and 189 text reasoning errors (Table 10).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenMathInstruct-1 data | — | distill-SFT | generator; prompt | Mixtral-base; 5-shot, instruction in Table 13 | arXiv:2402.10176v2 §2.1; HF dataset card | verified 2026-09-14 | chosen for math performance among open models and permissive license (§1); no generator ablation |
| OpenMathInstruct-1 data | — | distill-SFT | temperature; top_p; limits | 1.0; 0.95; 4,096 total tokens, ≤ 512 new tokens after each code block, ≤ 3 code blocks, stop at first execution error | v2 §2.1 | verified 2026-09-14 | no ablation reported |
| OpenMathInstruct-1 data | — | distill-SFT | samples per problem | GSM8K 128 per prompt, 256 total; MATH 224 per prompt, 896 total | v2 Table 2 | verified 2026-09-14 | Table 2 coverage per prompt |
| all OpenMath models | 7B–70B | SFT | examples | 512K fair-downsampled GSM8K + 511,677 Any-Code MATH ≈ 1.02M ("roughly 1.2M" in v1) | v2 §4, footnote 10 | verified 2026-09-14 | Tables 4, 6, 8 |
| all OpenMath models | 7B–70B | SFT | global batch; optimizer; weight decay; dropout | 128 (unit not stated); AdamW; 1e-2; 0.1 | v2 §3.1 | verified 2026-09-14 | no ablation reported |
| OpenMath-Mistral-7B | 7B | SFT | epochs; LR; GPUs (TP, PP) | 4; 1e-6; 64 (4, 1) | v2 App. B.1 Table 12 | verified 2026-09-14 | "based on our hyperparameter search" (B.1) |
| OpenMath-CodeLlama-7B / -13B | 7B, 13B | SFT | epochs; LR; GPUs (TP, PP) | 4; 2e-5; 64 (4, 1) | v2 Table 12 | verified 2026-09-14 | LR taken from ToRA (B.1) |
| OpenMath-CodeLlama-34B | 34B | SFT | epochs; LR; GPUs (TP, PP) | 4; 1e-5; 128 (8, 1) | v2 Table 12 | verified 2026-09-14 | no ablation reported |
| OpenMath-Llama2-70B | 70B | SFT | epochs; LR; GPUs (TP, PP) | 2; 1e-5; 256 (8, 2) | v2 Table 12 | verified 2026-09-14 | epochs limited "due to compute limitations" (B.1) |
| OpenMath-CodeLlama-70B | 70B | SFT | epochs; LR; GPUs (TP, PP) | 3; 1e-5; 256 (8, 2) | v2 Table 12 | verified 2026-09-14 | epochs limited "due to compute limitations" (B.1) |
| all OpenMath models | 7B–70B | SFT | checkpoint rule | 2 checkpoints per epoch (final runs); final checkpoint = average of all saved checkpoints | v2 §3.1 | verified 2026-09-14 | no ablation reported |
| all OpenMath models | 7B–70B | eval-gate | decoding | zero-shot; greedy; SC k=50 at temperature 0.7; generation continues after code errors | v2 §3.2, Table 3 | verified 2026-09-14 | T = 0.7 "beneficial" for majority voting (§3.2) |
| all OpenMath models | 7B–70B | SFT | LR schedule, warmup, sequence length, packing, prompt-token masking, GPU-hours | not reported | checked v2 §3.1, App. B.1, NeurIPS checklist, HF dataset card | not reported | — |

## Findings relevant to generality, negative feedback, distillation
- Generality: the authors state that in-domain benchmark gains may not transfer to related tasks, and that the drop from GSM8K to GSM-Hard (80.2 → 63.7 for OpenMath-Mistral-7B) indicates limited robustness to input perturbations (Limitations; Table 3). All problems come from GSM8K and MATH training sets (§2.1).
- Diversity (Interpretation, authors): subject-specific prompts lower greedy accuracy but raise SC accuracy (Table 7); code-preferential selection has the opposite effect (Table 8); SC gains over ToRA are attributed to the diversity of the fine-tuning data (§4).
- Negative samples: wrong-answer samples are discarded as training targets (negative marginal value, discard); the 6.6M incorrect solutions are released for verifier training and are not used in the paper (§1, footnote 4). Correct-answer solutions with flawed reasoning remain in the data (§2.3, Limitations).
- Distillation: data come from few-shot prompting of an open base model instead of a closed instruct model; the Mixtral-vs-GPT-4 coverage gap is the stated obstacle (§1, Fig. 1).

## Connections
- [[openmathinstruct-2]] — successor dataset by the same group.
- [[mammoth]], [[metamath]] — closed-model math SFT datasets that the abstract names as the motivation.
- [[numina-math]] — later tool-integrated competition-math dataset generated with GPT-4o.
- [[training-verifiers-to-solve-math-word-problems]] — source of GSM8K; the released incorrect solutions target verifier training of this kind.
- [[s1]], [[limo]] — small curated reasoning sets, in contrast to this sampling-heavy approach.

## Verification
- Checked on 2026-09-14 against: arXiv:2402.10176v2 (2024-11-03) including appendices and NeurIPS checklist; v1 (2024-02-15) for the corpus-size sentence; HF dataset card nvidia/OpenMathInstruct-1.
- Corrections to the previous card version: "teacher Mixtral-8x7B-Instruct-v0.1" → Mixtral-base (§2.1; card links Mixtral-8x7B-v0.1); "sample K=32–64 solutions per problem" → 128/256 (GSM8K) and 224/896 (MATH) (Table 2); "2–5 in-context examples" → K = 5 (§2.1); "~15K source questions, 7.5K GSM8K" → 7,473 GSM8K and 7,500 MATH training problems (§2.2.3); "OpenMath-Llama2-70B: 84.6 GSM8K, 50.7 MATH" → 84.7 / 46.3; 84.6 / 50.7 belong to OpenMath-CodeLlama-70B (Table 3); "OpenMath-Llama-70B" model name → OpenMath-Llama2-70B and OpenMath-CodeLlama-70B (Table 3); model family list completed with CodeLlama-7B/13B/34B (Table 12).
- Removed as unsupported by the source: ~500K GPU-hours of teacher sampling on DGX clusters; ~120 (GSM8K) and ~100 (MATH) solutions per problem; trace length 200–1,500 tokens, average ~500 tokens, 1–4 code blocks; SymPy canonicalization for MATH and string match for GSM8K as the grader; CoT-only ablation losing ~8 MATH points and PoT-only losing ~5 GSM8K points; 5–10% right-answer-wrong-reasoning rate (the paper calls such cases rare, §2.3); "Apache-2.0" as the dataset license (HF card: NVIDIA License permitting commercial use).
- Not reported by the source: answer-grading procedure details, generation compute, LR schedule and warmup, sequence length for SFT, packing.
