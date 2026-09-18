<!-- scope: OpenThoughts3 reasoning-SFT data recipe (question sourcing, mixing, filtering, answer multiplicity, answer filtering, teacher choice) and OpenThinker3-7B
     deps: [[sky-t1]], [[bespoke-stratos]], [[deepseek-r1]]
     see-also: [[s1]], [[limo]], [[openr1]], [[qwen-qwq-traces]], [[openmathinstruct-2]], [[numina-math]], [[packed-vs-unpacked-ablation]]
-->

# OpenThoughts: Data Recipes for Reasoning Models
- **Core Insight:** A reasoning-SFT data pipeline selected by more than 1,000 controlled 31,600-example experiments, scaled to 1.2M QwQ-32B-annotated examples, gives OpenThinker3-7B 53.3 on AIME25, 51.7 on LiveCodeBench 06/24–01/25, and 53.7 on GPQA-Diamond, 12.4 points above DeepSeek-R1-Distill-Qwen-7B on the 12-task average (Abstract; §1; Table 1).
- **Guideline:** When building a distillation-SFT set for math, code, and science reasoning at 7B scale, use one or two top-ranked question sources per domain, LLM-based question filters, and 16 teacher answers per question, and compare candidate teachers by training students on their outputs, because QwQ-32B produced better students than the higher-scoring DeepSeek-R1 in Table 8. Answer filtering gave no gain in §4.5 (Table 7).
- **Authors:** Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, et al.
- **Year:** 2025 (arXiv v1 2025-06)
- **URL:** https://arxiv.org/abs/2506.04178
- **Source type:** paper
- **Relevant topics:** reasoning distillation, SFT data curation, question sourcing and filtering, teacher selection, answer verification, decontamination, held-out evaluation

## Abstract
The OpenThoughts project builds open datasets for training reasoning models. OpenThoughts2-1M produced OpenThinker2-32B, described as the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks. The authors then study each step of the data-generation pipeline with more than 1,000 controlled experiments, which leads to OpenThoughts3. Scaling that pipeline to 1.2M examples with QwQ-32B as teacher yields OpenThinker3-7B: 53% on AIME 2025, 51% on LiveCodeBench 06/24–01/25, and 54% on GPQA Diamond, which are 15.3, 17.2, and 20.5 points above DeepSeek-R1-Distill-Qwen-7B. Datasets and models are released.

## Key Contributions
- A staged public data program: OpenThoughts-114K → OpenThoughts2-1M → OpenThoughts3-1.2M (§3, Table 2).
- A step-by-step ablation of the pipeline: question sourcing, mixing, question filtering, deduplication with multiple answers per question, answer filtering, teacher model (§4.1–4.6, Fig. 2).
- Five stated findings: 16× answer sampling as a scale axis; a better benchmark model is not necessarily a better teacher; no answer-filtering gain; top 1–2 sources beat mixing 8–16; LLM-based question filters beat embedding and fastText filters (§1).
- A held-out benchmark set (AIME25, HMMT 02/25, HLE MCQ subset, LiveCodeBench 06/24–01/25) that was not used for pipeline decisions (§4 "Evaluation Setup").

## Key Figures/Tables to Study
- **Table 1:** OpenThinker3-7B vs open-data 7B/8B reasoning models on 12 benchmarks.
- **Tables 3–8:** winners at each pipeline step (sources, mixing, question filters, dedup × sampling, answer filters, teachers).
- **Fig. 3:** scaling curves (316 to 31.6K examples) as each pipeline step is added.
- **Table 22:** effect of shortening reasoning traces. **Table 30:** safety and over-refusal. **Fig. 13:** robustness to problem perturbations.

## Technical Details
- **Earlier releases:** OpenThoughts-114K scales the Sky-T1 sources with DeepSeek-R1 answers verified by answer matching and unit tests: math 89K, code 20K, science 4K, puzzles 1K (§3). OpenThoughts2-1M adds sources selected by sweeping 26 question sources and uses 600K verified OpenR1-Math samples and 200K unverified math and code samples (§3).
- **Ablation protocol:** each strategy produces 31,600 examples (log-scale midpoint of 10K and 100K) and fine-tunes Qwen2.5-7B-Instruct; the strategy with the highest average over 8 development benchmarks is kept for the next step; DeepSeek-R1 is the teacher unless stated (§4).
- **Sourcing:** 27 code, 21 math, and 14 science question sources; GPT-4o-mini generates questions where an LLM is needed (§4.1). Best domain scores: CodeGolf 25.3 and OpenCodeReasoning 27.5 (code avg); OpenMath-2-Math 58.8 and NuminaMath-1.5 58.5 (math avg); StackExchange-Physics 43.2 and OrganicChemistry-PDF 45.3 (science avg); 17.2-point gap between best and worst code source (Table 3).
- **Mixing:** top-2 code sources average 41.3 vs 36.4 for top-16 (Table 4); at most two sources is best in all domains (§4.2).
- **Question filtering:** difficulty rating by GPT-4o-mini wins for code (43.0 avg); longest-response selection with GPT-4.1-mini wins for math (41.9 avg) and science; best filters add 4% (math) and 6% (code) over random filtering (§4.3, Table 5).
- **Dedup × answers:** 16 answers per question in all domains; exact dedup for math and science, none for code (§4.4, Table 6).
- **Answer filtering:** 63,200 answers are filtered to 31,600; for math, random filtering (64.8 math avg) beats every other filtering method, e.g. GPT verification 61.4; the no-filtering run (65.6) uses 63,200 examples and is not compute-controlled (§4.5, Table 7).
- **Teacher:** QwQ-32B vs DeepSeek-R1 average 44.2 vs 42.3 (code source) and 44.2 vs 41.6 (math source) (Table 8), although DeepSeek-R1 scores higher on CodeElo, GPQA-D, and JEEBench (§4.6). Appendix Tables 47–48 print different DeepSeek-R1 averages (38.0 code, 40.6 math); science: 39.1 vs 35.9 (Table 49).
- **Final mixture:** 850,000 math, 250,000 code, 100,000 science examples; ratio taken from OpenThoughts2-1M (§5). The OpenThoughts3-1.2M dataset card ("Data Curation and Scaling Recipe") lists: filter to 180k math / 60k code / 60k science questions, deduplicate, downsample to 75k questions, annotate each 16× with QwQ-32B.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenThinker3-7B | 7B | distill-SFT | base checkpoint | Qwen2.5-7B-Instruct | arXiv:2506.04178v2 §1, Table 1 | verified 2026-09-14 | App. G.1 Table 14 (100K examples): Qwen2.5-7B-Instruct scores higher than Llama-3.1-8B-Instruct overall |
| OpenThinker3-7B | 7B | distill-SFT | examples by domain | 850,000 math; 250,000 code; 100,000 science (1.2M) | §5 | verified 2026-09-14 | ratio copied from OpenThoughts2-1M (§5); no ratio ablation reported |
| OpenThinker3-7B | 7B | distill-SFT | teacher; answers per question | QwQ-32B; 16 | §4.6 Table 8; §4.4 | verified 2026-09-14 | Tables 8 and 6 (31,600-example runs, SEM reported) |
| OpenThinker3-7B | 7B | distill-SFT | question sources | OpenMath-2-Math; StackExchange CodeGolf + OpenCodeReasoning; StackExchange Physics + OrganicChemistry-PDFs | §4.2 takeaway | verified 2026-09-14 | Tables 3–4 |
| OpenThinker3-7B | 7B | distill-SFT | question filter; dedup; answer filter | GPT-4o-mini difficulty (code), GPT-4.1-mini response length (math, science); exact (math, science), none (code); none | §4.3–4.5 takeaways | verified 2026-09-14 | Tables 5, 6, 7 |
| OpenThinker3-7B | 7B | distill-SFT | teacher temperature, top_p, max tokens | not reported | checked §4, App. D, E, R.3, dataset card | not reported | App. E values are evaluation settings |
| OpenThinker3-7B | 7B | distill-SFT | LR; batch; epochs; packing | 8e-5; 512 (unit not stated); 5; yes | App. D.2 Table 9, "Large" set (> 31.6K) | derived (1.2M falls in the > 31.6K range) | per-scale sweeps mentioned, results not reported (App. D.2); packing check in Table 10 |
| OpenThinker3-7B | 7B | distill-SFT | schedule; optimizer; weight decay | cosine, warmup 0.1; AdamW β = (0.9, 0.999); 0 | App. D.2 | verified 2026-09-14 | no ablation reported |
| OpenThinker3-7B | 7B | distill-SFT | framework | LlamaFactory; DeepSpeed v3 without offloading | App. D.1–D.2 | verified 2026-09-14 | — |
| OpenThinker3-7B | 7B | distill-SFT | max sequence length | not reported (App. H.3 says SFT used traces "up to 16K tokens") | App. D, H.3 | not reported | — |
| OpenThinker3-7B | 7B | distill-SFT | compute | training 25,000 A100 GPU hours (128 nodes × 4 A100); annotation 22,000 H100 GPU hours | App. O | verified 2026-09-14 | — |
| Pipeline ablation students | 7B | distill-SFT | examples; LR; batch; epochs; packing | 31,600; 4e-5; 128; 5; no | §4 "Training"; App. D.2 Table 9 "Medium" (3.16K–31.6K) | derived (31,600 falls in Medium range) | 31,600 = log midpoint of 10K–100K (§4) |
| Pipeline ablation datasets | — | distill-SFT | default teacher; annotation cost | DeepSeek-R1; about $300 per 31,600-example dataset | §4 "Pipeline"; App. O | verified 2026-09-14 | — |
| All OpenThinker evals | — | eval-gate | temperature; top_p; max_new_tokens | 0.7; 1.0; 32,768 | App. E | verified 2026-09-14 | — |
| All OpenThinker evals | — | eval-gate | repeats | 10 (AIME24/25, AMC23, HMMT); 6 (LCB); 3 (CodeForces, CodeElo, GPQA-D, JEEBench, HLE); 1 (MATH500) | App. E | verified 2026-09-14 | — |
| Training data | — | eval-gate | decontamination | drop if Indel similarity reaches the 75% threshold or any 13-gram (Qwen2-7B-Instruct tokenizer) is shared; misses 12 of 3,092 planted items, drops 1.4% of 3,000 clean items | App. F | verified 2026-09-14 | tuned on a manually contaminated test set (App. F) |

## Findings relevant to generality, negative feedback, long context, distillation
- **Held-out and breadth:** OpenThinker3-7B has the best HMMT, AIME25, and LCB 06/24–01/25 scores among compared open-data 7B/8B models (§5), but on HLE MCQ it scores 10.2 vs 12.7 for Qwen2.5-7B-Instruct (Table 1), and HLE does not respond to data scale (App. G).
- **Cross-domain transfer:** code-only sets differ on GPQA-D (47.3 vs 36.7), but both mixes with science data reach 52.7 (App. H.5, Table 24). The authors note that selecting by overall average assumes cross-domain transfer (§6).
- **Robustness:** all tested distilled reasoning models, including OpenThinker3-7B, show large accuracy changes across structure-preserving Alice-in-Wonderland problem variants (App. N, Fig. 13).
- **Safety narrowing:** HarmBench harmfulness rate rises from 14.5 (Qwen2.5-7B-Instruct) to 55.5 (OpenThinker3-7B); XSTest over-refusal 4.4 → 5.6; no safety data was used (App. L, Table 30).
- **Filtering out negatives (negative marginal value):** verification on OpenThoughts-114K lowers the 7B average (45.0 unverified → 41.9) and raises the 32B average (62.1 → 64.5) (App. H.1.1, Table 15); LLM-generated unit-test filtering on 16,000 code examples gives LCB 36.0 vs 38.5 unfiltered (Table 18).
- **Trace length:** removing self-reflection phrases cuts average trace length from 11,593 to 328 tokens and the average score from 51.4 to 26.3 (−49.1%); dropping traces over 2,048 tokens gives 34.2 (−33.3%) (App. H.3, Table 22).
- **Distillation:** Claude 3.7 annotations were slightly worse than DeepSeek-R1 for code and science (Tables 19–20); OpenThinker3-7B exceeds its teacher QwQ-32B on JEEBench, 72.4 vs 69.9 (App. K, Table 29); on a Lawma legal task, Qwen2.5-7B students (0.819–0.828) exceed the DeepSeek-R1 annotator (0.739) (App. J, Table 28).

## Connections
- [[sky-t1]] and [[bespoke-stratos]] — the project's starting pipeline and first dataset (§3, App. C).
- [[deepseek-r1]] — default teacher for ablations and teacher of OpenThoughts-114K.
- [[qwen-qwq-traces]] — QwQ-32B, the teacher chosen for OpenThoughts3.
- [[s1]] and [[limo]] — small curated datasets used as scaling baselines in Fig. 1.
- [[openr1]] — OpenR1-Math is a source for OpenThoughts2-1M; OpenR1-Distill-7B is a Table 1 baseline.
- [[openmathinstruct-2]] and [[numina-math]] — the top two math question sources in Table 3.
- [[packed-vs-unpacked-ablation]] — contrasts with App. D.3, where packing did not lower scores.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2506.04178 (arXiv v2, 5 Jun 2025; body and App. A–S) and https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M (dataset card)
- Corrections to the previous card version:
  - "sampling multiple answers per question is the easiest way to expand a source by at least 16x" → "an effective technique to increase the size of a data source by at least 16×" (§1).
  - "none beat training on all answers without filtering" → no answer filter beat the baseline; the no-filtering run is not compute-controlled (63,200 vs 31,600 examples) (§4.5).
  - "[[s1]]: both argue that curation and test-time reasoning budget matter more than raw scale" → OpenThoughts reports scores rising with dataset size up to 1M examples (§5, Fig. 3; App. G, Fig. 8); s1 is a Fig. 1 baseline.
  - Authors "Etash Guha et al." → first six authors listed.
- Removed as unsupported by the source: "Open-data advantage is fragile: later gains depend on keeping the entire pipeline open"; "the paper repeatedly shows that some cleanup steps do not pay for the lost scale" (replaced by §4.5 numbers); "extends [[limo]] and [[lima]]" (LIMA is not discussed).
- Not reported by the source: teacher sampling temperature/top_p/length, maximum training sequence length, number of optimizer steps. The earlier audit's attribution of temperature 0.7 / top_p 1.0 / 32,768 tokens to the teacher is incorrect; App. E gives these for evaluation.
