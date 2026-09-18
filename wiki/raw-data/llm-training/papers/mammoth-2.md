<!-- scope: MAmmoTH2 / WebInstruct — 10M instruction-response pairs recalled, extracted, and LLM-refined from Common Crawl; SFT results on four base models; MAmmoTH2-Plus
     deps: [[mammoth]]
     see-also: [[deepseekmath]], [[metamath]], [[openhermes-2-5]], [[openmathinstruct]], [[mathscale]], [[openmathinstruct-2]]
-->

# MAmmoTH2: Scaling Instructions from the Web
- **Core Insight:** A recall → extract → refine pipeline over Common Crawl produces WebInstruct, 10M instruction-response pairs (about 5B tokens), and SFT on WebInstruct alone raises Mistral-7B from 11.2 to 36.7 on MATH and from 36.2 to 68.4 on GSM8K, although WebInstruct contains no training data from these benchmarks (Table 1, Table 2, arXiv v4).
- **Guideline:** When the available reasoning SFT data is seeded only from GSM8K/MATH-style problems and broad STEM reasoning is the target, use naturally occurring web Q-A pairs that are recalled, extracted, and refined by LLMs, because WebInstruct-only SFT improved all seven held-out reasoning benchmarks on Mistral-7B, Llama-3-8B, Mixtral-8×7B, and Yi-34B (Table 2).
- **Authors:** Xiang Yue, Tuney Zheng, Ge Zhang, Wenhu Chen (Carnegie Mellon University; University of Waterloo)
- **Year:** 2024 (arXiv v1 2024-05; v4 2024-05-23)
- **URL:** https://arxiv.org/abs/2405.03548
- **Source type:** paper
- **Relevant topics:** web-mined instruction data, reasoning SFT, data recall classifiers, LLM refinement, decontamination, domain mixture

## Abstract
Instruction data is usually produced by human crowd-sourcing or GPT-4 distillation. The paper instead harvests 10 million naturally occurring instruction-response pairs from the pre-training web corpus in three steps: recall relevant documents, extract Q-A pairs, and refine the pairs with open-source LLMs. Base models fine-tuned on this dataset (MAmmoTH2) improve on reasoning benchmarks; MAmmoTH2-7B (Mistral) rises from 11% to 36.7% on MATH and from 36% to 68.4% on GSM8K without in-domain training data. Further training on public instruction datasets gives MAmmoTH2-Plus, which the authors report as state of the art on several reasoning and chat benchmarks.

## Key Contributions
- WebInstruct: 10M Q-A pairs mined from Common Crawl, about 5B tokens (§2, Table 1).
- A three-stage pipeline: two-round fastText recall with LLM domain selection, LLM extraction, and refinement by two LLMs (§2.1–§2.3, Fig. 3).
- MAmmoTH2 models on Mistral-7B, Llama-3-8B, Mixtral-8×7B, and Yi-34B, and MAmmoTH2-Plus after continued training on public datasets (§3.1, §2.5).
- Ablations on instruction count and loss type, refining LLM, domain and source subsets, plus a 50-sample human quality check (§5, Fig. 5, Tables 4–5, Fig. 6).

## Key Figures/Tables to Study
- **Figure 3 / Figure 4** — pipeline and one extraction → refinement example.
- **Table 1** — WebInstruct vs SFT datasets (pairs) and continued-training corpora (tokens).
- **Table 2** — seven reasoning benchmarks for all MAmmoTH2 and MAmmoTH2-Plus models.
- **Table 3** — code, MT-Bench, AlpacaEval 2.0, Arena Hard, MMLU, MMLU-Pro.
- **Table 7 (App. E)** — WebInstruct only vs public datasets only vs both.

## Technical Details
- **Recall seed (§2.1):** 100K exam problems crawled from educational websites as positives and 100K random Common Crawl documents as negatives train a fastText model (vector dimension 256, 3 epochs, learning rate 0.1, maximum n-gram length 3, "maximum number of word occurrences of 3").
- **Recall round 1 (§2.1):** 100B tokens recalled from an internal Common Crawl; documents grouped by root URL, domains with more than 1000 documents kept (about 600K domains); GPT-3.5 selects about 50K domains as likely to contain instruction data. Round-1 documents are not used downstream.
- **Recall round 2 (§2.1):** fastText is retrained with documents from selected domains as positives and documents from non-selected domains and general Common Crawl as negatives; it recalls 40B tokens; GPT-4 filters the recalled domains again, giving 18M raw documents.
- **Extraction (§2.2):** rule-based HTML pre-processing removes site information, ads, and boilerplate; Qwen-72B with few-shot examples extracts Q-A pairs or returns void. 30% of documents contain Q-A pairs, giving about 5M candidates.
- **Decontamination (§2.2):** web pages with a 10-gram string match to any question or answer in the evaluation benchmarks are removed.
- **Refinement (§2.3):** Mixtral-22B×8 and Qwen-72B reformat the pairs and add intermediate reasoning steps when the answer has none; using two models is intended to increase diversity; output is 10M pairs.
- **Internal inconsistency:** §1 names Mixtral as the extractor, Mixtral-8×7B as a refiner, and GPT-4 for root-URL trimming; §2.1–§2.3 name Qwen-72B (extraction), Mixtral-22B×8 (refinement), and GPT-3.5 then GPT-4 (domain selection). The section text is more detailed.
- **Composition (App. G):** subject labels by Llama-3-8B-Instruct: 81.69% "Science", within which "Mathematics takes up the largest share at 68.36%"; 86.73% exam-style (education) and 13.27% forum by source URL.
- **Human check (§5.4, Fig. 6):** of 50 random refined samples, 78% were improved over the extracted version and 10% had hallucinations introduced by refinement.
- **Plus-stage data (§2.5, App. A):** OpenHermes 2.5 (1M; TheoremQA removed from Platypus), Code-Feedback (68K multi-turn), and Math-Plus (894K: MetaMathQA 395K, Orca-Math 200K, and 300K GPT-4 rewrites of MATH training Q-A). Training order: WebInstruct first, then the public datasets (App. E).
- **Main results (Table 2, few-shot CoT):** MAmmoTH2-7B: TheoremQA 29.0, MATH 36.7, GSM8K 68.4, GPQA 32.4, MMLU-STEM 62.4, BBH 58.6, ARC-C 81.7, average 52.8 (+14.0 over Mistral-7B). Average gains over base: MAmmoTH2-8B +8.8, MAmmoTH2-8x7B +6.5, MAmmoTH2-34B +5.8.
- **Plus results (Table 2):** MAmmoTH2-7B-Plus MATH 45.0, GSM8K 84.7, average 58.0; MAmmoTH2-8B-Plus average 59.1 vs Llama-3-8B-Instruct 53.4; MAmmoTH2-8x7B-Plus average 62.9 vs Qwen-1.5-110B 63.6. MATH and GSM8K are not held-out for Plus models (Table 2 row headers).
- **General benchmarks (Table 3):** MAmmoTH2-8x7B-Plus AlpacaEval 2.0 33.8, Arena Hard 32.6, MMLU-Pro 50.4 (GPT-3.5-Turbo-1106: 19.3 and 18.9 on AlpacaEval 2.0 and Arena Hard). MAmmoTH2-7B-Plus code average 66.1 over HumanEval and MBPP (58.2 over HumanEval+ and MBPP+).
- **Refiner ablation (§5.2, Table 4; Mistral-7B, 9000 steps, global batch 512):** GSM/MATH/MMLU-S/TheoremQA/ARC = Mixtral 62.9/29.1/56.5/26.1/78.3; Qwen 65.4/28.9/60.6/23.5/80.8; merged 65.6/31.0/60.5/24.8/81.8.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| MAmmoTH2-7B / 8B / 8x7B / 34B | 7B, 8B, 8×7B, 34B | SFT | data | WebInstruct, 10M pairs, ~5B tokens | arXiv:2405.03548v4 §2.3, Table 1 | verified 2026-09-14 | Fig. 5: MATH, TheoremQA, ARC-C accuracy increases with instruction count up to 10M (Mistral-7B; no numbers in text) |
| MAmmoTH2-7B (Mistral) | 7B | SFT | peak LR | 5e-6 | §3.1 | verified 2026-09-14 | no ablation reported |
| MAmmoTH2-8B, -8x7B, -34B | 8B, 8×7B, 34B | SFT | peak LR | 1e-5 | §3.1 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH2 | all | SFT | global batch; max sequence length | 512 (unit not stated); 4096 | §3.1 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH2 | all | SFT | schedule; epochs | cosine, 3% warm-up; 2 epochs | §3.1 | verified 2026-09-14 | no ablation reported |
| all MAmmoTH2 | all | SFT | framework; compute | LLaMA-Factory, DeepSpeed ZeRO-3; 32 A100 GPUs (GPU-hours not reported) | §3.1 | verified 2026-09-14 | — |
| all MAmmoTH2 | all | SFT | loss for the main runs | not reported (§5.1 compares "LM loss" and "SFT loss" without defining them) | checked §3.1, §5.1, Fig. 5 | not reported | Fig. 5: SFT loss on refined QA above LM loss (Mistral-7B; no numbers in text) |
| all MAmmoTH2 | all | SFT | optimizer, weight decay, grad clip, packing | not reported | checked §3.1, §5, App. A–J | not reported | — |
| MAmmoTH2-7B-Plus / 8B-Plus / 8x7B-Plus | 7B, 8B, 8×7B | SFT (continued) | data; order | OpenHermes 2.5 1M + Code-Feedback 68K + Math-Plus 894K, after WebInstruct | §2.5, App. A, App. E | verified 2026-09-14 | Table 7: Mistral-7B avg 58.0 (both) vs 52.8 (WebInstruct) vs 53.4 (public only) |
| MAmmoTH2-*-Plus | all Plus | SFT (continued) | LR, batch, epochs for this stage | not reported separately | checked §3.1, App. A, App. E | not reported | — |

## Findings relevant to generality and distillation
- **Held-out evaluation (Result, single study):** WebInstruct-only models are evaluated on benchmarks removed from the data by 10-gram decontamination (§2.2, Table 2). The authors read the Table 3 code and chat results as evidence that the method "does not overfit to the reasoning benchmarks" (§4.2, Interpretation).
- **Comparison with an official instruct model:** MAmmoTH2-8B-Plus is higher than Llama-3-8B-Instruct on the Table 2 reasoning average (59.1 vs 53.4) and on MMLU-Pro (43.4 vs 40.9), but lower on code (61.9 vs 65.8), MT-Bench (7.95 vs 8.02), AlpacaEval 2.0 (18.5 vs 22.9), Arena Hard (16.6 vs 20.6), and MMLU (64.6 vs 67.2) (Table 3). The paper describes this as "matching its performance on general tasks" (§1).
- **Domain narrowing (§5.3, Table 5; Mistral-7B subsets):** the math-domain subset gives the highest MATH (27.3) but GSM8K 52.9 and MMLU-STEM 51.6 against 47.4 and 51.4 for the "Base" row; the "Other" subset gives the highest GSM8K (59.4) and the science subset the highest ARC-C (83.6). Education data beats forum data on GSM8K, MATH, TheoremQA, and ARC-C, not on MMLU-STEM (54.3 vs 54.7).
- **Coverage limits (App. H):** humanities and daily chat topics may be underrepresented; extraction and refinement errors depend on the LLMs used.
- **Distillation:** refinement asks open LLMs to write missing reasoning steps; 10% of 50 human-checked samples gained hallucinations (§5.4). Merged Mixtral + Qwen refinements scored highest on GSM8K, MATH, and ARC-C, but not on MMLU-STEM or TheoremQA (Table 4), although the §5.2 text says the merged model "consistently outperforms".

## Connections
- [[mammoth]] — earlier work from the same group; MathInstruct (262K) is listed in Table 1 as a GPT-4-synthesized comparator.
- [[deepseekmath]] — recall-from-Common-Crawl predecessor (120B tokens, Table 1); decontamination follows it (§2.2).
- [[metamath]], [[openhermes-2-5]] — MetaMathQA and OpenHermes 2.5 are part of the MAmmoTH2-Plus data (App. A).
- [[openmathinstruct]] — 1.8M Mixtral-synthesized GSM/MATH-seeded comparator in Table 1.
- [[mathscale]] — cited in §1 among approaches that prompt GPT-4 with seed data.
- [[openmathinstruct-2]] — a later math SFT dataset; not discussed in this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2405.03548 (arXiv v4, 2024-05-23, full text with Appendices A–J); v1 and the project page (tiger-ai-lab.github.io/MAmmoTH2) compared for version differences.
- Version note: v1 and the project page report MAmmoTH2-7B MATH 34% (v1 Table 2: 34.2) and GSM8K 67% (67.4); v1 §2.1 uses GPT-4 in both domain-selection rounds, and the project page names Mixtral-8×7B as the extractor. This card uses v4.
- Corrections to the previous card version:
  - Recall "fastText + DeBERTa-v3 classifier applied to ~18B documents; seed ~100K StackExchange/Socratic pages" → fastText only, two rounds (100B then 40B tokens recalled), seeds are 100K crawled exam problems, 18M documents after GPT-4 domain filtering (§2.1).
  - Extraction "regex/DOM rules on ~3M pages + Mixtral-8x7B; ~11M raw pairs" → rule-based HTML cleaning, then Qwen-72B extraction; ~5M candidates (§2.2).
  - Refinement "Mixtral-8x7B; LLM judge 1–5, keep ≥ 4"; "Teacher: Mixtral-8x7B-Instruct" → Mixtral-22B×8 and Qwen-72B; no judge-score filter reported (§2.3).
  - "MATH +22, ARC-C +9"; "MATH 11 → 32.6, GSM8K 51.4 → 67.4, ARC-C 80.5 → 84.7" → MATH 11.2 → 36.7 (+25.5), GSM8K 36.2 → 68.4, ARC-C 74.2 → 81.7 (+7.5) (Table 2).
  - "MAmmoTH2-Plus-8x7B: MATH 48.4, TheoremQA 34.1 — competitive with GPT-3.5-Turbo" → MATH 47.0, TheoremQA 34.1 (Table 2); the GPT-3.5-Turbo comparison is on AlpacaEval 2.0 and Arena Hard (Table 3).
  - "Mixing WEBINSTRUCT with MathInstruct gives +3 MATH" → the Plus stage uses OpenHermes 2.5, Code-Feedback, and Math-Plus; on Mistral-7B, MATH is 45.0 with both vs 37.9 public-only and 36.7 WebInstruct-only (Table 7).
  - "~8% error rate" → 78% improved and 10% hallucination introduced in a 50-sample human check (Fig. 6).
  - "Domain mix ~40% math/physics, ~20% CS/code, ~20% STEM, ~20% humanities" → 81.69% Science, Mathematics 68.36%, 86.73% education vs 13.27% forum (App. G).
  - "underrepresentation of non-English" → App. H names humanities and daily chat topics; language coverage is not discussed.
- Removed as unsupported by the source: "~200K GPU-hours"; "median ~300 tokens, tail to 3K"; "MinHash dedup on question text"; "authors argue scale compensates for noise"; "web-mined data generalizes better OOD than seed-synthesized data" (no controlled comparison); "research-only license" (the project page lists a partial dataset and a full dataset marked "Approval Required"); "cannot be regenerated without CC access"; connection claims about [[rephrasing-the-web]] and [[fineweb]] ("FineWeb-Edu uses similar quality classifier"; neither is discussed in the paper).
- Not reported by the source: reasoning-length distribution; answer-correctness verification; optimizer settings; Plus-stage hyperparameters; GPU-hours.
