<!-- scope: NuminaMath technical report (July 2024) — NuminaMath-CoT (860K) and NuminaMath-TIR datasets, decontamination, and the two-stage SFT recipe behind the 1st AIMO Progress Prize solution
     deps: [[mammoth]]
     see-also: [[openmathinstruct-2]], [[openr1]], [[s1]], [[limo]]
-->

# NuminaMath: The largest public dataset in AI4Maths with 860k pairs of competition math problems and solutions
- **Core Insight:** A 7B model (DeepSeekMath-Base 7B) fine-tuned first on the 860K-pair NuminaMath-CoT set and then on GPT-4o-generated tool-integrated reasoning (TIR) data reaches 68.1% on MATH and 20/40 on AMC 2023 (0-shot) with TIR, against 55.8% and 11/40 for the CoT-only stage (§4.3, Table 3); the TIR model won the 1st AIMO Progress Prize (Abstract; NuminaMath-7B-TIR model card).
- **Guideline:** When assembling a competition-math SFT set from web and PDF sources that overlap with evaluation suites, decontaminate with both n-gram matching and embedding nearest-neighbour search, because MATH test problems include reworded AIME/AMC problems that exact string matching misses (§3.4).
- **Authors:** Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, et al. (Numina, Hugging Face, MIT, Mistral AI, Peking University, Answer AI)
- **Year:** 2024 (report dated July 22, 2024; not on arXiv)
- **URL:** https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf ; datasets https://huggingface.co/datasets/AI-MO/NuminaMath-CoT , https://huggingface.co/datasets/AI-MO/NuminaMath-TIR
- **Source type:** official technical report (with dataset cards and released training configs)
- **Relevant topics:** competition math, CoT realignment, tool-integrated reasoning, decontamination, open math datasets, SFT

## Abstract
Numina presents NuminaMath, a collection of 860,000 competition-math problem–solution pairs ranging from high-school exercises to advanced competition problems, each with a chain-of-thought (CoT) solution. The report describes how the data were collected and processed, the dataset composition, and experiments showing its effect. A model fine-tuned on the dataset won the 1st AIMO Progress Prize. The authors describe it as the largest math dataset released to date.

## Key Contributions
- NuminaMath-CoT: 860K problem–solution pairs, of which 400K are competition-level (§3, Table 1).
- A processing pipeline: OCR (Mathpix), regex or manual segmentation, and GPT-4o translation, CoT realignment, and answer boxing (§3.2).
- NuminaMath-TIR: solutions interleaving text and executed Python, sampled from GPT-4o and filtered by answer match (§3.3).
- A two-step decontamination procedure: 10-gram exact match plus Mistral-embedding distance < 0.15 (§3.4).
- A two-stage SFT recipe (CoT, then TIR) at 7B and 72B, plus self-consistent TIR decoding (SC-TIR) (§4.1, §4.2.1).

## Key Figures/Tables to Study
- Table 1 (dataset sizes), Figure 3 (samples per source), Table 2 (hyperparameters), Tables 3–4 (7B and 72B results), Algorithms 1–2 (TIR and SC-TIR).

## Technical Details
**Sources and per-source processing (§3.1)**
- MATH and GSM8K: GPT-4 reformats the reference solutions into the CoT format, following DeepSeekMath and ReAlign.
- Orca-Math: answers are located with regular expressions and wrapped in `\boxed{}`; the solutions are not rewritten.
- AMC and AIME: statements from the AoPS wiki; the first community solution containing `\boxed{}` is kept; about 6,500 problems, about 4,300 retained after embedding decontamination; CoT realignment by GPT-4o.
- AoPS forum: problems crawled from the Contest Collection page; among replies with `\boxed{}` or ■, the one with the most LaTeX symbols becomes the reference and is rewritten by GPT-4o.
- Chinese K-12 exams: public exam papers; OCR and regex segmentation for PDFs; GPT-4o translates and realigns solutions. The report states these are not competition level.
- Synthetic data: following Li et al. (2024), GPT-4 samples new problems from MATH and AMC-AIME training seeds at temperature 0.8; Numina keeps the first-stage solution instead of a second greedy solution, to reduce cost.
- World olympiads: 152K pairs from international contests and shortlists, national and regional contests, forums, olympiad books, and summer-school material.
- Released source counts (NuminaMath-CoT card, rev 9d8d210): cn_k12 276,591; synthetic_math 167,895; orca_math 153,334; olympiads 150,581; synthetic_amc 62,111; aops_forum 30,201; math 7,478; gsm8k 7,345; amc_aime 4,072; table total 859,608. Split metadata lists 859,494 train and 100 test rows.

**TIR data (§3.3)**
1. Select about 100K problems with value (numeric) answers from NuminaMath-CoT.
2. Sample one solution per problem with the GPT-4o assistant API at temperature 0.8, using the ToRA prompt.
3. Discard samples whose answer does not match the reference: exact match for integer answers, GPT-4o as judge for other expressions.
4. Repeat the sampling on the problems whose samples were discarded.
- The dataset card (rev 77a91d7) instead states about 70k problems, GPT-4 as generator, and three repetitions; it lists 72,441 train and 99 test rows.

**Decontamination (§3.4)**
- Evaluation sets protected: MATH, GSM8K, AIME 2024, AMC12 2023.
- Step 1: remove 10-gram exact matches from all subsets except the synthetic set and MATH train. MATH-test problems that paraphrase MATH-train problems with different numbers are not flagged.
- Step 2: embed every problem (except MATH train and synthetic) with Mistral embeddings and remove problems at distance < 0.15, a threshold "derived empirically, above which we did not observe contamination in our internal tests".

**Inference (§4.2)**
- TIR loop: sample text until the stop string ```` ```output ````; if `\boxed{}` is present, parse and return the answer; if no ```` ```python ```` block is present, discard the text and resample; otherwise execute the code and append its output or a truncated traceback; stop after k rounds and return an error value (for example −1 on AIMO) (Algorithm 1).
- SC-TIR: sample n TIR trajectories, filter answers outside the valid set, and take the majority vote (Algorithm 2).

**Results (§4.3)**
- 7B, Table 3: CoT → TIR: GSM8K 76.3% → 84.6%; MATH 55.8% → 68.1%; AMC 2023 0-shot 11/40 → 20/40, maj@64 18/40 → 31/40; AIME 2024 0-shot 0/30 → 5/30, maj@64 1/30 → 10/30. DeepSeekMath 7B RL: 88.2%, 51.7%, 9/40, 14/40, 1/30, 1/30.
- 72B, Table 4: CoT → TIR: GSM8K 91.4% → 91.5%; MATH 68.0% → 75.8%; AMC 2023 0-shot 21/40 → 24/40, maj@64 24/40 → 34/40; AIME 2024 0-shot 1/30 → 5/30, maj@64 3/30 → 12/30. GPT-4o 0513: 95.8%, 76.6%, 20/40, 2/30 (0-shot).
- Only the NuminaMath TIR columns use tool-integrated reasoning; all other scores are without TIR (Table 3 and Table 4 captions).
- AIMO score: 29/50 on the public and private test sets (NuminaMath-7B-TIR model card, rev cf2aaf3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| NuminaMath-7B-CoT / -TIR | 7B | SFT | base checkpoint | DeepSeekMath-Base 7B | report §4.1; stage-1-cot.yaml@a3f4d5d | verified 2026-09-14 | no ablation reported |
| NuminaMath-72B CoT / TIR | 72B | SFT | base checkpoint | Qwen2-72B | report §4.1 | verified 2026-09-14 | no ablation reported |
| all NuminaMath models | 7B, 72B | SFT | global batch; peak LR; schedule | 32; 2 × 10⁻⁵; cosine | report §4.1 | verified 2026-09-14 | no ablation reported |
| NuminaMath-7B-CoT | 7B | SFT | batch derivation in config | 4 per device × 8 processes × 1 grad-accum = 32 | stage-1-cot.yaml + deepspeed_zero3.yaml@a3f4d5d | derived | — |
| NuminaMath-7B-CoT | 7B | SFT | block size; epochs; warmup ratio | 2048; 3; 0 | report Table 2; stage-1-cot.yaml@a3f4d5d | verified 2026-09-14 | no ablation reported |
| NuminaMath-7B-TIR | 7B | SFT | block size; epochs; warmup ratio | 1024; 4; 0.1 | report Table 2; stage-2-tir.yaml@a3f4d5d | verified 2026-09-14 | no ablation reported |
| NuminaMath-72B CoT | 72B | SFT | block size; warmup ratio | 2048; 0.1 | report Table 2 | verified 2026-09-14 | no ablation reported |
| NuminaMath-72B CoT | 72B | SFT | epochs | Table 2: 1; §4.1 text: "only trained for 2 epochs in Stage 1" | report Table 2 vs §4.1 | conflict | shortened "due to time constraints" (§4.1) |
| NuminaMath-72B TIR | 72B | SFT | block size; epochs; warmup ratio | 2048; 3; 0.1 | report Table 2 | verified 2026-09-14 | no ablation reported |
| NuminaMath-7B-CoT / -TIR | 7B | SFT | precision; parallelism | bf16; DeepSpeed ZeRO-3 | stage-1-cot.yaml, deepspeed_zero3.yaml@a3f4d5d | verified 2026-09-14 | — |
| all NuminaMath models | 7B, 72B | SFT | compute | 8–32 H100 by model size; < 24 h for both stages | report §4.1 | verified 2026-09-14 | — |
| Kaggle solution models | 7B | SFT | compute | one node of 8 × H100, 10 hours | GitHub README "Numina Solution…" @a3f4d5d | verified 2026-09-14 | — |
| NuminaMath-CoT data | — | SFT | examples | 859,494 train / 100 test | HF dataset card metadata rev 9d8d210 | verified 2026-09-14 | — |
| NuminaMath-TIR data | — | distill-SFT | generator; problems; rounds | GPT-4o, T = 0.8, ~100K problems, repeat on failures (report) vs GPT-4, ~70k problems, 3 repeats (dataset card) | report §3.3 vs TIR card rev 77a91d7 | conflict | filter: answer match vs reference (§3.3) |
| NuminaMath-TIR data | — | distill-SFT | examples | 72,441 train / 99 test | TIR card metadata rev 77a91d7 | verified 2026-09-14 | — |
| Kaggle solution | 7B | eval-gate | checkpoint selection | four validation sets from AMC, AIME, and MATH | GitHub README @a3f4d5d | verified 2026-09-14 | — |
| all | — | SFT | weight decay, loss masking, packing, tokens | not reported | checked: report §4.1, Table 2; stage configs@a3f4d5d | not reported | — |

## Findings relevant to generality, negative feedback, distillation
- Narrowing: "Models fine-tuned on the NuminaMath dataset may lose some capacity for general instruction following"; proposed remedies are rebalancing the data and new alignment objectives, not measured (§5). The 7B TIR model card states the model "should not be used for general chat applications" and often fails on AIME-level, olympiad, and geometry problems (model card, "Bias, Risks, and Limitations").
- Measurement: canary strings are included in the dataset to detect training contamination (§5); about 40% of the dataset is proof-based problems without a checkable final answer (§5).
- Negative samples: TIR samples with a wrong final answer are discarded and the problem is resampled (negative marginal value, discard) (§3.3). Synthetic problems and solutions come from one sampling stage at temperature 0.8 without a correctness check; §5 calls their correctness "challenging to validate" and names GPT-4o, while §3.1 names GPT-4 for the same step.
- Distillation: the dataset "heavily relies on GPT-4o"; moving to open models is listed as future work (§5).

## Connections
- [[mammoth]] — cited as prior work that trains on both CoT and program trajectories; ToRA and MuMath-Code are named as the direct basis of Numina's TIR strategy (§2).
- [[s1]] — NuminaMATH supplies 30,660 of s1's 59K candidate questions (arXiv:2501.19393 App. C, Table 7).
- [[limo]] — uses NuminaMath-CoT in its candidate pool (§3.1.1); Qwen2.5-32B-Instruct fine-tuned on a random 100k NuminaMath-CoT subset averages 32.3% pass@1 over 10 benchmarks vs 49.9% for the base model (arXiv:2502.03387 §6.2, Table 1).
- [[openr1]] — OpenR1-Math-220k uses problems from NuminaMath 1.5, a later release than this report (OpenR1-Math-220k dataset card).
- [[openmathinstruct-2]] — another teacher-generated math SFT corpus compared in ch-24.

## Verification
- Checked on 2026-09-14 against: numina_dataset.pdf (report dated July 22, 2024); github.com/project-numina/aimo-progress-prize README and training/configs at commit a3f4d5d; HF dataset cards AI-MO/NuminaMath-CoT (rev 9d8d210) and AI-MO/NuminaMath-TIR (rev 77a91d7); AI-MO/NuminaMath-7B-TIR model card (rev cf2aaf3).
- Corrections to the previous card version: "cn_k12 ~470K" → 276,591; "orca-math ~200K" → 153,334 in the release (200K is the original Orca-Math size, §3); source list omitted synthetic_math 167,895 and synthetic_amc 62,111; "CoT split: GPT-4o 2-shot generation kept if SymPy-equivalent" → reference solutions are reformatted/realigned by GPT-4o (GPT-4 for MATH/GSM8K), not regenerated and filtered (§3.1–3.2); "TIR: execute code, keep if executed answer matches gold" → exact match for integers, GPT-4o judge otherwise (§3.3); "MinHash dedup across splits" → 10-gram exact match plus embedding distance < 0.15 (§3.4); "some samples from DeepSeek-Math-7B in a bootstrapping loop" → not in report; "Decontamination is the user's responsibility" → the report performs decontamination (§3.4); "Apache-2.0, 8+ sources" → Apache-2.0 confirmed, 9 source tags; "transfers well to AMC12, AIME, OMCA" → results reported on GSM8K, MATH, AMC 2023, AIME 2024 only.
- Removed as unsupported by the source: SymPy verifier; CoT ~400 and TIR ~600 average/median tokens; 1–3 code blocks per TIR solution; ~$100K+ API cost; "OpenR1-Math-220k is a NuminaMath-CoT subset" (it uses NuminaMath 1.5); "Sky-T1 uses it in the math mix" (not checked against a Sky-T1 primary source); translation-error and "TIR false positive" risk statements; "GPT-4o fails top-10% of AIME/IMO" statement; [[mathscale]] lineage clause.
- Not reported by the source: token statistics as numbers (only figures), weight decay, packing, loss masking, 72B per-stage wall-clock time.
