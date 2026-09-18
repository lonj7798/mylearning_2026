<!-- scope: Bespoke Labs blog (Jan 2025) on Bespoke-Stratos-32B/7B, SFT on Bespoke-Stratos-17k (16,710 examples) built with the Sky-T1 data pipeline and DeepSeek-R1 as teacher; plus the official dataset/model cards and curation script it links
     deps: [[deepseek-r1]], [[sky-t1]]
     see-also: [[openr1]], [[s1]], [[limo]], [[qwen-qwq-traces]], [[deepseek-r1-distill-synth]]
-->

# Bespoke-Stratos: The unreasonable effectiveness of reasoning distillation
- **Core Insight:** Bespoke-Stratos-32B, an SFT of Qwen2.5-32B-Instruct on the 16,710-example Bespoke-Stratos-17k dataset, scores AIME2024 63.3 / MATH500 93.0 / GPQA-Diamond 58.1 / LiveCodeBench v2 All 71.1, against 66.7 / 89.8 / 61.1 / 72.2 for DeepSeek-R1-Distill-Qwen-32B in the authors' own evaluation, with 47x fewer training examples (blog, 32B table).
- **Guideline:** When an answer parser rejects many correct teacher solutions, replace or supplement it with a model judge before rejection sampling, because Bespoke Labs report that swapping Sky-T1's regex+sympy parser for gpt-4o-mini raised the share of retained correct math solutions from 25% to 73% (blog, "Data Curation").
- **Authors:** Bespoke Labs (organization; the citation block names no individuals)
- **Year:** 2025 (blog dated 2025-01-22)
- **URL:** https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation
- **Source type:** official blog (with official dataset card, model cards, and curation code linked from it)
- **Relevant topics:** reasoning distillation, distill-SFT, rejection sampling, verifier false negatives, small-data SFT

## Summary
Bespoke Labs trained Bespoke-Stratos-32B by distilling DeepSeek-R1 with the Sky-T1 data pipeline ported into their Curator library. The blog reports that the model outperforms Sky-T1-32B and o1-preview on the listed math and code benchmarks and comes close to DeepSeek-R1-Distill-Qwen-32B while using 47x fewer examples. It also releases Bespoke-Stratos-7B (a fine-tune of Qwen2.5-7B-Instruct), the dataset, and the curation code. Three changes to the Sky-T1 recipe are listed: DeepSeek-R1 replaces QwQ as teacher, R1 traces are not reformatted by gpt-4o-mini, and gpt-4o-mini replaces Sky-T1's parsing logic for filtering incorrect math solutions.

## Key Contributions
- Bespoke-Stratos-17k dataset (Apache-2.0; 16,710 rows in the HF viewer) and the curation scripts.
- Bespoke-Stratos-32B and -7B SFT checkpoints with benchmark tables against Sky-T1, o1-preview, and R1-Distill models.
- A measured verifier fix: model-judge filtering of math answers instead of regex+sympy parsing (25% → 73% retained correct solutions).
- A 7B result that contrasts with the Sky-T1 authors' note of little or no improvement at 7B/14B with their data.

## Key Figures/Tables to Study
- Blog 32B table: AIME2024, MATH500, GPQA-Diamond, LiveCodeBench v2 Easy/Medium/Hard/All.
- Blog 7B table: Bespoke-Stratos-7B vs Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-7B (ours/reported).
- HF model cards, "Training hyperparameters" section.

## Technical Details
- **Teacher and pipeline:** DeepSeek-R1 as teacher instead of QwQ; Sky-T1 pipeline ported into Curator; generation took 1.5 hours (blog, "Data Curation"); cost $800 (dataset card, "Details").
- **Prompt composition, dataset card:** 5k coding (APPs, TACO), 10k math (AIME, MATH, Olympiads subsets of NuminaMATH), 1k science and puzzle (STILL-2); exact problems may differ because of rejection sampling (dataset card, "Details").
- **Prompt composition, curation README:** Numina 10.5k (math, olympiads, amc_aime subsets of the difficulty-labeled Numina set), APPS ~2.5k, TACO ~3k, STILL-2 ~1k (curator repo `examples/bespoke-stratos-data-generation/README.md`).
- **Math filter:** gpt-4o-mini judges the R1 solution against the ground-truth solution; Sky-T1's regex+sympy parsing "often fails to extract the right answer" and filtered correct solutions; retained correct solutions rose from 25% to 73% (blog, "Data Curation"). The script implements a structured `correct: bool` judge with `model_name="gpt-4o-mini"` (`generate_numina_data.py` L17-43, L111-119).
- **Code filter:** rejection sampling by code execution, sped up with a Ray cluster (blog, "Data Curation"); APPS rows pass through `process_dataset_parallel` and are kept if `correctness` is true (`generate_apps_data.py` L64; `combine_data.py` L63).
- **Formatting:** R1 traces were not reformatted (blog). The released script wraps each trace as `<|begin_of_thought|> … <|end_of_thought|> <|begin_of_solution|> … <|end_of_solution|>` under the Sky-T1 system prompt; math prompts are prefixed "Return your final response within \boxed{}." (`combine_data.py` L9-21).
- **Generation parameters in the released script:** `model_name="deepseek-reasoner"`, `generation_params={"temp": 0.0}` for Numina (`generate_numina_data.py` L76-83); the APPS script sets no generation parameters (L44-51). Olympiads prompts are capped with `.take(20_000)` before generation (L93-95); correct Numina rows are capped at 10,500 (`combine_data.py` L24-26).
- **STILL-2 rows in the released script:** taken from `RUC-AIBOX/long_form_thought_data_5k`, filtered to domains puzzle, physics, biology, chemistry, using the dataset's existing `combined_text` as the assistant turn; the script applies no R1 call or verifier to these rows (`combine_data.py` L116-129).
- **Students:** Qwen2.5-32B-Instruct and Qwen2.5-7B-Instruct, full fine-tune (dataset card; model card tags "llama-factory", "full").

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | training examples | 16,710 rows (train split) | HF dataset card bespokelabs/Bespoke-Stratos-17k, viewer "Number of rows" | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | prompt mix (examples) | 10k math / 5k code / 1k science+puzzle | dataset card "Details" | verified 2026-09-14 | card states the mix is similar to Sky-T1_data_17k; no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | teacher; reformatting | DeepSeek-R1; no reformatting | blog "Data Curation" | verified 2026-09-14 | no ablation reported (authors state R1 traces were well-formatted and coherent without reformatting) |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | math correctness filter | gpt-4o-mini judge vs ground truth | blog "Data Curation" | verified 2026-09-14 | retained correct solutions 25% (parser) → 73% (judge) |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | teacher sampling temperature (official posts) | not reported | checked blog, dataset card, both model cards | not reported | — |
| Bespoke-Stratos-17k curation script | — | distill-SFT | `generation_params` passed to deepseek-reasoner (Numina subsets) | `{"temp": 0.0}` | github.com/bespokelabsai/curator main, `examples/bespoke-stratos-data-generation/generate_numina_data.py` L78 | verified 2026-09-14 (released code; not confirmed as the run that produced the dataset) | no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | peak LR; schedule; warmup | 1e-05; cosine; warmup ratio 0.1 | HF model cards, "Training hyperparameters" | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | global batch (sequences) | 96 = 1 per device × 8 devices × 12 grad-accum | HF model cards, "Training hyperparameters" | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | optimizer | adamw_torch, betas (0.9, 0.999), eps 1e-08; seed 42 | HF model cards | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | epochs | 3.0 | HF model cards | verified 2026-09-14 | no ablation reported |
| Bespoke-Stratos-32B | 32B | distill-SFT | compute | 8×H100, 27 hours | HF model card 32B, "Training procedure" | verified 2026-09-14 | n/a |
| Bespoke-Stratos-7B | 7B | distill-SFT | compute | 8×H100, 7 hours | HF model card 7B, "Training procedure" | verified 2026-09-14 | n/a |
| Bespoke-Stratos-32B, -7B | 32B, 7B | distill-SFT | max sequence length; packing; loss masking; weight decay | not reported | checked blog, dataset card, both model cards | not reported | — |

## Findings relevant to distillation and negative feedback
- **Distillation gap, 32B (Result, single study):** Bespoke-Stratos-32B vs DeepSeek-R1-Distill-Qwen-32B (ours / reported): AIME2024 63.3 vs 66.7 / 72.6; MATH500 93.0 vs 89.8 / 94.3; GPQA-Diamond 58.1 vs 61.1 / 62.1; LCB v2 All 71.1 vs 72.2 / not reported; LCB v2 Hard 26.2 vs 38.2 (blog, 32B table). Sky-T1-32B: 43.3 / 82.4 / 56.8 / 57.9.
- **Distillation gap, 7B (Result, single study):** Bespoke-Stratos-7B vs Qwen2.5-7B-Instruct vs R1-Distill-Qwen-7B (ours / reported): AIME2024 20.0 / 10.0 / 43.3 / 55.5; MATH500 82.0 / 74.2 / 89.4 / 92.8; GPQA-Diamond 37.8 / 33.3 / 44.9 / 49.1; LCB v2 All 36.1 / 31.9 / 46.6 (blog, 7B table). LCB v2 Hard is 1.6, below the instruct base's 3.3.
- **Data scale:** the blog attributes the 7B gain, compared with Sky-T1's reported lack of gain, possibly to higher data quality (Interpretation); the 7B model card attributes the remaining gap to R1-Distill possibly to 17k vs 800k examples (Interpretation).
- **Negatives as discards:** incorrect traces are discarded (negative marginal value, no gradient use). The reported failure mode is verifier false negatives: the parser discarded correct solutions, and the judge reduced them (blog, "Data Curation").
- **Generality:** only math, code, and GPQA-Diamond are evaluated; the authors state that "benchmarks convey only one side to the story" and invite evaluation on other benchmarks (blog, "Thoughts and future work"). Evaluation sampling settings are not given; the blog refers to the Sky-T1 codebase.

## Connections
- [[sky-t1]]: the data pipeline and prompt mix that Bespoke-Stratos ports and modifies.
- [[deepseek-r1]]: the teacher model; its R1-Distill-Qwen checkpoints are the comparison baselines.
- [[qwen-qwq-traces]]: QwQ, the teacher that Sky-T1 used and Bespoke-Stratos replaced.
- [[openr1]], [[deepseek-r1-distill-synth]]: other R1-trace distillation efforts.
- [[s1]], [[limo]]: other small-example reasoning SFT datasets.

## Verification
- Checked on 2026-09-14 against: the blog URL above (dated 2025-01-22); https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k (dataset card); https://huggingface.co/bespokelabs/Bespoke-Stratos-32B and https://huggingface.co/bespokelabs/Bespoke-Stratos-7B (model cards); https://github.com/bespokelabsai/curator/tree/main/examples/bespoke-stratos-data-generation (main branch; no commit hash shown on the fetched page).
- Corrections to the previous card version:
  - Title "Bespoke-Stratos: A small, open replication of R1 distillation" → exact blog title above. Authors "Mahesh Sathiamoorthy, et al." → "Bespoke Labs" (citation block).
  - Prompt mix "math ~7K (NuminaMath-CoT, MATH, AIME/AMC), code ~5K (APPS, CodeContests, TACO, LeetCode), science ~5K (STILL-2, CoTLogic)" → 10k math / 5k code (APPs, TACO) / 1k science+puzzle (STILL-2) (dataset card).
  - "Math verified by SymPy canonicalization" → gpt-4o-mini judge; regex+sympy parsing was the replaced Sky-T1 method (blog).
  - "Science verified by GPT-4o LLM judge" → no science judge in blog or cards; the released script uses STILL-2 rows without a verifier (`combine_data.py` L116-129).
  - "`<think>`/`<answer>` format preserved" → `<|begin_of_thought|>`/`<|begin_of_solution|>` wrappers (`combine_data.py` L12-14).
  - "Stratos-32B LiveCodeBench ~57%, within 2-3 points of R1-Distill-Qwen-32B" → LCB v2 All 71.1 (57.9 is Sky-T1-32B); AIME2024 63.3 vs 72.6 reported / 66.7 own eval.
  - "~50× more data-efficient" → "47x fewer examples" (blog). "Release of Curator" → Curator was used; the curation code was released.
  - "$800 disclosed in blog" → $800 is in the dataset card; the blog gives 1.5 hours. "Training ~$4,000 on 8×H100 for a few hours" → 8×H100 for 27 hours (32B) and 7 hours (7B); no dollar figure.
- Removed as unsupported by the source: temperature 0.6; 3× retries; MinHash dedup and per-source caps; trace-length distribution (1K–10K, median ~3K); 30–50% rejection rate; ablations "removing code verification halves LCB gain" and "removing symbolic equivalence halves MATH gain"; "matched Sky-T1 at 1/5 compute"; "90%+ of gain for 1/50 data cost"; the claim that the authors cite [[s1]]/[[limo]]; "~671B params"; the "Risks + gotchas" list (not reported by the source); the [[self-instruct]] lineage claim.
- Not reported by the source: max sequence length, packing, loss masking, evaluation sampling settings, contamination checks.
