<!-- scope: Magicoder / OSS-Instruct (UIUC, Dec 2023; ICML 2024) — code instruction data written by gpt-3.5-turbo-1106 from random 1-15-line open-source seed snippets; 75K samples fine-tune CodeLlama-Python-7B and DeepSeek-Coder-Base-6.7B
     deps: [[self-instruct]]
     see-also: [[evol-instruct]], [[code-evol-instruct]], [[wizardcoder]], [[opc-synthetic-code]], [[phi-textbooks]]
-->

# Magicoder: Empowering Code Generation with OSS-Instruct
- **Core Insight:** Fine-tuning CodeLlama-Python-7B for 2 epochs on 75K problem-solution pairs that a teacher wrote from random open-source code snippets raises HumanEval+ pass@1 from 34.1 to 55.5, whereas 75K comment-function pairs mined from the same seed documents leave it at 34.1 (§4.2, Table 6); continuing on the 110K-sample evol-codealpaca-v1 gives 66.5, above GPT-3.5 Turbo's 65.9 (Table 1).
- **Guideline:** When an open-source code corpus is the starting material for code SFT data, use its snippets as seeds for a teacher that writes self-contained problems and solutions instead of training on mined comment-function pairs, because in the paper's 7B ablation the mined pairs gave no HumanEval+ gain and lowered MultiPL-E from 29.6 to 24.1 (§4.2, Table 6).
- **Authors:** Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, Lingming Zhang (University of Illinois Urbana-Champaign; Zhe Wang: Tsinghua University)
- **Year:** 2023 (arXiv v1 2023-12, titled "Magicoder: Source Code Is All You Need"; v2 2024-06; ICML 2024, PMLR 235)
- **URL:** https://arxiv.org/abs/2312.02120
- **Source type:** paper
- **Relevant topics:** synthetic code instructions, seed-grounded data generation, OSS-Instruct, Magicoder, decontamination, code SFT, weaker-teacher distillation

## Abstract
Magicoder is a series of fully open-source (code, weights, data) code LLMs with at most 7B parameters, trained on 75K synthetic instruction samples produced by OSS-Instruct. OSS-Instruct prompts an LLM with open-source code snippets to generate diverse coding instructions; the stated motivation is to reduce the bias of LLM-generated synthetic data by grounding it in open-source references. Because OSS-Instruct is orthogonal to methods such as Evol-Instruct, the authors also build MagicoderS by combining the two. Both series outperform code models of similar or larger size on a range of coding benchmarks, and MagicoderS-CL-7B, based on CodeLlama, exceeds ChatGPT on HumanEval+ (66.5 vs 65.9 pass@1).

## Key Contributions
- OSS-Instruct: a teacher LLM receives a random seed snippet from open-source code and writes a coding problem and its solution (§2.1, App. A.1).
- Magicoder (75K OSS-Instruct) and MagicoderS (Magicoder further fine-tuned on evol-codealpaca-v1) on CodeLlama-Python-7B (CL) and DeepSeek-Coder-Base-6.7B (DS) (§3).
- Evaluation on HumanEval(+), MBPP(+), MultiPL-E, DS-1000, APPS (§3, App. C).
- Ablations on training-language mix (§4.1), direct fine-tuning on the seed code (§4.2), a weaker open teacher (§4.3), and removal of noisy samples (App. C.3).

## Key Figures/Tables to Study
- **Figures 1-2, 5:** seed snippet → problem → solution examples (shell script, imports, class signature, comments).
- **Figure 3:** TF-IDF similarity of generated data to HumanEval, compared with Code Alpaca and evol-codealpaca-v1.
- **Table 1 / Table 4:** pass@1 for the CL and DS series. **Tables 5-7, 10:** ablations.

## Technical Details
- Seed corpus: starcoderdata, the filtered, permissively licensed version of The Stack used to train StarCoder, already post-processed for decontamination (§2.1).
- Seeds: 1-15 consecutive lines extracted at random from each document; 80K seeds from 80K documents, 40K Python and 5K each of C++, Java, TypeScript, Shell, C#, Rust, PHP, Swift (§2.1).
- Prompt (Fig. 4): "Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in two distinct sections: [Problem Description] and [Solution]." The problem must be "completely self-contained"; the solution "comprehensive, correct".
- Teacher: gpt-3.5-turbo-1106 with greedy decoding, chosen to "maximize the consistency between the generated problems and solutions" (App. B.1).
- Cleaning: samples that are identical or share the same seed are removed; samples with other noise such as incomplete solutions are kept (§2.2). No execution or test-based check of solutions is described (§2.2, App. B).
- Decontamination follows StarCoder: problems containing HumanEval or MBPP docstrings or solutions, APPS docstrings, DS-1000 prompts, or GSM8K questions are removed (§2.2); 9 OSS-Instruct samples and 89 evol-codealpaca-v1 samples were removed (App. B.2). The final dataset has about 75K entries (§2.2).
- Language mix of the output: about 43K Python-only and 32K non-Python samples, split by whether "```python" appears (§4.1); Python is "around 57%" (§4.1).
- Similarity to HumanEval (Fig. 3): mean of each sample's highest TF-IDF cosine similarity to the 164 HumanEval problems is 0.105 for OSS-Instruct, 0.131 for evol-codealpaca-v1, 0.169 for Code Alpaca (Self-Instruct, 20K).
- Categories: embeddings (INSTRUCTOR) against 10 manually designed coding categories; the authors describe the distribution as diverse and balanced (App. A.3, Fig. 6) and give no numeric comparison with other datasets.
- HumanEval (+) / MBPP (+) pass@1, greedy, from the EvalPlus leaderboard (Table 1): CodeLlama-Python-7B 37.8 (34.1) / 57.6 (45.4); WizardCoder-CL-7B 48.2 (40.9) / 56.6 (47.1); Magicoder-CL-7B 60.4 (55.5) / 64.2 (52.6); MagicoderS-CL-7B 70.7 (66.5) / 68.4 (56.6); GPT-3.5 Turbo 72.6 (65.9) / 81.7 (69.4).
- DS series (Table 4): Magicoder-DS-6.7B 66.5 (60.4) / 75.4 (61.9); MagicoderS-DS-6.7B 76.8 (70.7) / 75.7 (64.4); DeepSeek-Coder-Instruct-6.7B 73.8 (70.1) / 72.7 (63.4) with "×8 fewer" fine-tuning tokens for MagicoderS-DS (§3.4).
- DS-1000 completion overall (Table 3): Magicoder-CL 29.9, MagicoderS-CL 37.5, WizardCoder-SC-15B 29.2.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Magicoder-CL-7B, Magicoder-DS-6.7B | 7B, 6.7B | distill-SFT | examples; teacher | about 75K OSS-Instruct samples; gpt-3.5-turbo-1106, greedy decoding | arXiv:2312.02120v2 §2.2, App. B.1, B.3 | verified 2026-09-14 | Table 6: 55.5 vs 34.1 (comment-function pairs) HumanEval+, CL-7B |
| Magicoder-CL-7B, Magicoder-DS-6.7B | 7B, 6.7B | distill-SFT | epochs | 2 | App. B.3 | verified 2026-09-14 | no ablation reported |
| Magicoder-CL-7B, Magicoder-DS-6.7B | 7B, 6.7B | distill-SFT | LR; warmup; schedule | initial LR 5e-5; 15 warmup steps; linear scheduler | App. B.3 | verified 2026-09-14 | no ablation reported |
| Magicoder-CL-7B, Magicoder-DS-6.7B | 7B, 6.7B | distill-SFT | optimizer; batch; length | Adafactor; batch size 512 (unit not stated); sequence truncation length 1216 | App. B.3 | verified 2026-09-14 | no ablation reported |
| Magicoder-CL-7B, Magicoder-DS-6.7B | 7B, 6.7B | distill-SFT | compute | 2 A100-80GB, PyTorch DDP, Hugging Face transformers | App. B.3 | verified 2026-09-14 | — |
| Magicoder-DS-6.7B | 6.7B | distill-SFT | fine-tuning tokens | +90M | Table 4 | verified 2026-09-14 | — |
| MagicoderS-CL-7B, MagicoderS-DS-6.7B | 7B, 6.7B | distill-SFT (stage 2) | examples | evol-codealpaca-v1, about 110K samples, open-source Evol-Instruct reproduction generated by GPT-4 | §3, App. B.2-B.3 | verified 2026-09-14 | Table 1: 66.5 vs 55.5 HumanEval+ (CL-7B) |
| MagicoderS-CL-7B, MagicoderS-DS-6.7B | 7B, 6.7B | distill-SFT (stage 2) | other settings | "the same hyperparameters except for 15 warmup steps and a 1024 maximum sequence length" (as printed; 15 warmup steps is also the stage-1 value) | App. B.3 | verified 2026-09-14 | no ablation reported |
| MagicoderS-DS-6.7B | 6.7B | distill-SFT (both stages) | fine-tuning tokens | +240M | Table 4 | verified 2026-09-14 | — |
| Ablation runs (CodeLlama-Python-7B) | 7B | distill-SFT | epochs; other settings | 2 epochs (Tables 5-7, 10); Tables 5-6 state the App. B hyperparameters, Tables 7 and 10 do not restate them | §4.1-4.3; App. C.3 | verified 2026-09-14 | — |

## Findings relevant to generality, negative samples, and distillation
- **Cross-language transfer (Result, single study; Table 5, CL-7B, 2 epochs).** HumanEval+ / MultiPL-E average: base 34.1 / 29.6; Python-only 43K 47.6 / 32.7; non-Python 32K 44.5 / 38.3; both 75K 55.5 / 37.8. Non-Python data alone raises Python HumanEval+ by 10.4 points (§4.1). The authors attribute the lower multilingual score of "both" vs non-Python to the Python share of about 57% (§4.1).
- **Format vs content (§4.2, Table 6).** Comment-function pairs have a format close to HumanEval and MultiPL-E but gave 34.1 HumanEval+ and 24.1 MultiPL-E (base 29.6). The authors conjecture that noise and inconsistency in mined pairs cause the drop and conclude that "data factuality, rather than the format", matters (Interpretation).
- **Benchmark overlap check.** Generated data are the least similar to HumanEval of the three datasets compared (Fig. 3), which the authors use to argue that gains are not from same-distribution data (§2.3).
- **Negative marginal value of noisy samples (App. C.3, Table 10).** Removing samples with partial implementations (pass or NotImplemented tokens) reduces 75K to 68K and HumanEval+ from 55.5 to 54.9; the authors keep noisy samples. One run per setting is reported.
- **Weaker-teacher distillation (§4.3, Table 7).** CL-7B fine-tuned for 2 epochs on 20K OSS-Instruct samples from Mixtral-8x7B-Instruct-v0.1 reaches 55.5 HumanEval+ and 50.4 MBPP+, above the teacher's 39.6 and 47.4. The authors interpret this as the seed snippets and the base model's pre-trained code ability contributing beyond teacher distillation (Interpretation).
- **Harder problems (App. C.1, Table 8).** On 300 APPS problems, MagicoderS-CL-7B scores 8.7 overall (base 2.3) and 1.7 on the 60 competition problems.
- Not reported: evaluation on non-code tasks, so retention of general ability after code SFT is not measured.

## Connections
- [[self-instruct]] — Code Alpaca (20K samples from 21 seed tasks) applies Self-Instruct; §1 describes it as relying on a narrow set of predefined tasks.
- [[code-evol-instruct]] / [[wizardcoder]] — Code Evol-Instruct uses 5 heuristics on Code Alpaca seeds (§1); MagicoderS uses the open reproduction evol-codealpaca-v1 because the official dataset was not released (§2.3).
- [[evol-instruct]] — the general-instruction method that Code Evol-Instruct adapts.
- [[opc-synthetic-code]] — its card records a Magicoder citation for seed-snippet synthesis in stage 2.
- [[phi-textbooks]] — cited in §5 as textbook-quality synthetic data for code.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2312.02120 (v2, ICML 2024); v1 read for the title and Table 1 values.
- Corrections to the previous card version:
  - Title "Magicoder / OSS-Instruct: Empowering Code Generation with Open-Source Code Seeds" → "Magicoder: Empowering Code Generation with OSS-Instruct" (v2); v1 was "Magicoder: Source Code Is All You Need".
  - "(UIUC)" for all authors → Zhe Wang is affiliated with Tsinghua University (front matter).
  - "WizardCoder-CL-7B 51.8" → 40.9 HumanEval+ (Table 1); 51.8 is CodeLlama-Python-34B on HumanEval.
  - "MBPP+ 49.3" → MagicoderS-CL-7B MBPP+ is 56.6 in Table 1 of both v1 and v2.
  - "75K–100K instructions sufficient to close the gap to ChatGPT on HumanEval+" → 75K OSS-Instruct alone gives 55.5 vs 65.9; exceeding ChatGPT required the additional 110K evol-codealpaca-v1 stage (Table 1).
  - "CodeLlama-7B base" → CodeLlama-Python-7B; DeepSeek-Coder-Base-6.7B variants added (§3).
  - "Teacher GPT-3.5-Turbo" → gpt-3.5-turbo-1106 with greedy decoding (App. B.1).
  - "decontamination (n-gram overlap); dedup; language/format validation" → identical/same-seed removal and StarCoder-style decontamination against five benchmarks; no format validation; noisy samples kept (§2.2, App. B.2).
  - "pipeline emphasizes Python" → seeds are 40K Python and 40K across 8 other languages (§2.1).
  - Prompt "sketch" → exact prompt text from Fig. 4.
  - "OSS-Instruct + Evol-Instruct combine additively" → continued fine-tuning improves results (Table 1); no Evol-Instruct-only run on the same base, so additivity is not tested.
- Removed as unsupported by the source: "mean-response attractor" framing; post-processing into "Question:/Answer:" format; "MIT license"; "a few hundred USD" cost; lexical-diversity comparison with Self-Instruct/Evol-Instruct; "category distribution more balanced than Code-Evol-Instruct"; GPL license-leakage risk (the seed corpus is permissively licensed, §2.1); "teacher sometimes ignores the snippet"; "later studies found residual overlap"; "SOTA among open 7B at release"; "Dataset used in many open models"; link claim to [[prismatic-synthesis]] about gradient coverage (not in this paper).
- Not reported by the source: API cost; execution-based verification of solutions; license of the released dataset.
