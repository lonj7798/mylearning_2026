<!-- scope: self-supervised API-call annotation of raw text, filtered by whether the tool result lowers future-token loss
     deps: [[self-instruct]]
     see-also: [[toolllm]], [[apigen]], [[gorilla]], [[toolace]], [[ccnet]]
-->

# Toolformer: Language Models Can Teach Themselves to Use Tools
- **Core Insight:** Sampling candidate API calls inside plain text, executing them, and keeping only the calls whose returned result lowers the weighted cross-entropy on following tokens by at least τ_f produces tool-use supervision with no human labels; fine-tuning GPT-J (6.7B) on the resulting corpus raises SQuAD-subset LAMA accuracy from 17.8 to 33.8 and MAWPS from 9.9 to 44.0, above GPT-3 (175B) at 26.8 and 19.8 (§4.2.1 Table 3, §4.2.2 Table 4).
- **Guideline:** When generating tool-call supervision without human labels, execute each candidate call and accept it only if the returned result reduces loss on subsequent tokens relative to both no call and the call without its result, because the second comparison is what prevents accepting calls that only match a surface pattern (§2).
- **Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, et al. (Meta AI Research; Universitat Pompeu Fabra)
- **Year:** 2023 (arXiv v1 2023-02)
- **URL:** https://arxiv.org/abs/2302.04761
- **Source type:** paper
- **Relevant topics:** tool use, function calling, self-supervised annotation, synthetic data, loss-based filtering

## Abstract
Toolformer is a language model trained to decide which APIs to call, when to call them, what arguments to pass, and how to use the results in later token prediction. The training data is built in a self-supervised way from a handful of demonstrations per API: the base model proposes API calls inline in raw text, the calls are executed, and only the calls whose results reduce future-token loss are kept. The filtered corpus preserves the original text and adds the accepted calls. The resulting 6.7B model improves zero-shot performance across several downstream tasks, in some cases above much larger models, without degrading language-modeling perplexity.

## Key Contributions
- A self-supervised annotation procedure that turns unlabeled text into tool-call supervision from a few demonstrations per API (§2).
- A usefulness filter defined on the model's own loss rather than on schema validity or human labels (§2).
- Five text-in/text-out tools evaluated together: question answering, Wikipedia search, calculator, calendar, machine translation (§3).
- Evidence that tool-augmented 6.7B predictions beat OPT (66B) and GPT-3 (175B) on LAMA subsets and on three math benchmarks (Tables 3, 4).
- A model-size study locating the point where tool use starts to help (§4.4).

## Key Figures/Tables to Study
- **Figure 1** — examples of the model deciding to call QA, calculator, MT, and Wikipedia search.
- **Table 2** — number of accepted API-call examples per tool at τ_f ∈ {0.5, 1.0, 2.0}.
- **Tables 3-7** — zero-shot results on LAMA, math, QA, MLQA, and temporal datasets.
- **Table 8** — WikiText and CCNet perplexity, to check the method does not degrade language modeling.
- **Figure 4** — performance with and without API calls across GPT-2 sizes and GPT-J.

## Technical Details
**Base setup.** Language model M is GPT-J with 6.7B parameters; the corpus C is a subset of CCNet (§4.1). Five tools: QA (Atlas-large during data generation, Atlas-xxl at inference), a Python calculator supporting +, −, *, / rounded to two decimals, a BM25 Wikipedia retriever over the KILT dump, the 600M NLLB translation model with fastText source-language detection and English as target, and a calendar returning the current date (§3, App. A.1).

**Annotation pipeline (§2).**
1. Prompt M with a few demonstrations per API so it inserts `<API>` … `</API>` → result spans inline.
2. Compute p_i = p_M(`<API>` | P(x), x_{1:i−1}) for each position i; keep positions with p_i > τ_s, at most the top k.
3. Sample up to m candidate calls per kept position and execute them.
4. Accept a call when L⁻_i − L⁺_i ≥ τ_f, where L⁺_i is the weighted cross-entropy on following tokens with the call and its result given as prefix, and L⁻_i is the minimum of the loss with no call at all and the loss with the call but no result.
5. Interleave accepted calls with the original text to form C*, then fine-tune M on C* with the standard LM objective. Apart from the inserted calls, C* contains exactly the same text as C.

**Thresholds (App. A).** Defaults are τ_s = 0.05 (the `<API>` token must have at least 5% probability), τ_f = 1.0, k = 5, m = 5. For the calculator and machine translation tools, heuristic pre-filtering shrinks the candidate pool, so the paper sets τ_s = 0.0, k = 20, m = 10, and τ_f = 0.5 for those two tools.

**Position weighting (§4.1).** The loss weights are w_t = w̃_t / Σ_s w̃_s with w̃_t = max(0, 1 − 0.2·t), where t indexes tokens after the call position. This concentrates credit on tokens near the call.

**Heuristic candidate narrowing (§4.1, App. A.1).** Calculator candidates require at least three numbers in the text. For machine translation, only paragraphs containing a 10-token chunk classified as non-English by fastText with confidence above 0.8, preceded and followed by English text, are kept; chunks that are only numbers or symbols are dropped, and calls whose MT input appears only after the call are removed because the model cannot look ahead at inference.

**Yield (Table 2).** At τ_f = 1.0 the accepted example counts are Wikipedia Search 60,974, Calendar 20,587, QA 18,526, MT 1,034, Calculator 994. At τ_f = 0.5 the counts are 207,241 / 61,811 / 51,987 / 3,156 / 3,680; at τ_f = 2.0 they are 13,944 / 3,007 / 5,135 / 229 / 138. The paper states the method is sample-inefficient: processing more than a million documents yields only a few thousand useful calls for some tools (§7).

**Inference (§2, §4.2).** Decoding proceeds normally until `→` is produced, at which point the API is called and both the response and `</API>` are inserted. For evaluation the model may start a call when `<API>` is among the k = 10 most likely tokens, and at most one API call per input is allowed.

**Results.** LAMA subsets, accuracy (Table 3): Toolformer 33.8 / 11.5 / 53.5 on SQuAD / Google-RE / T-REx against GPT-J 17.8 / 4.9 / 31.9, OPT (66B) 21.6 / 2.9 / 30.1, GPT-3 (175B) 26.8 / 7.0 / 39.8; the QA tool is used for 98.1% of examples. Math (Table 4): Toolformer 40.4 ASDiv / 29.4 SVAMP / 44.0 MAWPS against GPT-3 (175B) 14.0 / 10.0 / 19.8, with the calculator used on 97.9% of examples. QA (Table 5): Toolformer 26.3 WebQS / 17.7 NQ / 48.8 TriviaQA, above GPT-J but below GPT-3 (175B) at 29.0 / 22.6 / 65.9, with Wikipedia search used on 99.3% of examples.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Toolformer (GPT-J base) | 6.7B | distill-SFT (self-annotated) | training corpus | subset of CCNet, augmented to C* with accepted API calls | arXiv:2302.04761v1 §4.1 | verified 2026-09-18 | no ablation reported |
| Toolformer | 6.7B | distill-SFT | batch size (sequences) | 128 | §4.1 | verified 2026-09-18 | no ablation reported |
| Toolformer | 6.7B | distill-SFT | learning rate | 1·10⁻⁵, linear warmup over the first 10% of training | §4.1 | verified 2026-09-18 | no ablation reported |
| Toolformer | 6.7B | data generation | sampling threshold τ_s | 0.05 (0.0 for calculator and MT) | App. A | verified 2026-09-18 | chosen per tool "to ensure a sufficiently larger number of examples" (§4.1); no ablation reported |
| Toolformer | 6.7B | data generation | filtering threshold τ_f | 1.0 (0.5 for calculator and MT) | App. A; yields at 0.5/1.0/2.0 in Table 2 | verified 2026-09-18 | Table 2 reports yield per threshold, not downstream accuracy per threshold |
| Toolformer | 6.7B | data generation | candidate positions k / calls per position m | 5 / 5 (20 / 10 for calculator and MT) | App. A | verified 2026-09-18 | no ablation reported |
| Toolformer | 6.7B | eval-gate | decoding: `<API>` allowed when in top-k; max calls per input | k = 10; 1 call | §4.2 | verified 2026-09-18 | §5 "Decoding Strategy" varies k (Table 9) |
| Toolformer | 6.7B | distill-SFT | epochs, optimizer, sequence length, compute | not reported | body §4.1, App. B referenced but no values printed in the paper text | not reported | — |

## Findings relevant to generality and distillation
- Language modeling is preserved: WikiText perplexity is 9.9 for GPT-J, 10.3 for GPT-J + CC, and 10.3 for Toolformer with API calls disabled; CCNet validation perplexity is 10.6 / 10.5 / 10.5 (§4.3, Table 8). The paper notes it cannot evaluate perplexity with API calls enabled because that would require marginalizing over all potential calls (§4.3, footnote 8).
- Tool use does not emerge at small scale: applying the method to GPT-2 models at 124M, 355M, 775M and 1.6B shows tools only begin to help at around 775M parameters, with Wikipedia search the exception since the paper judges that API easier to use (§4.4, Figure 4). The gap between predictions with and without calls stays large at 6.7B.
- Part of the gain is not from calling tools at all: on the math benchmarks "Toolformer (disabled)" already beats GPT-J (14.8 / 6.3 / 15.0 vs 7.5 / 5.2 / 9.9), which the authors attribute to fine-tuning on many examples of calls and their results (§4.2.2, Interpretation).
- Multilingual transfer is not consistent: on MLQA, API calls improve Toolformer over its own disabled variant for all six languages, but continued pretraining on CCNet degrades performance enough that Toolformer does not consistently beat vanilla GPT-J; the MT tool is used on 63.8-94.9% of examples except Hindi at 7.3% (§4.2.4, Table 6).
- Attribution caution the paper applies to itself: the TEMPLAMA gain is not from the calendar tool, which is used on 0.2% of examples, but from Wikipedia search and QA; on DATESET the calendar tool is used on 54.8% and the gain is attributed to it (§4.2.5).
- Stated limitations (§7): no chained tool use, because calls for each tool are sampled independently and no chained examples exist in C*; no interactive use such as reformulating a search query or browsing multiple results; sensitivity to the exact wording of the input when deciding whether to call an API; sample inefficiency.

## Connections
- [[self-instruct]] shares the pattern of converting a few seed demonstrations into a larger synthetic training set, with a model-side filter instead of human labels.
- [[ccnet]] is the corpus this method annotates.
- [[toolllm]] moves from single inline calls to multi-step trajectories over real APIs.
- [[apigen]] and [[toolace]] replace the self-annotation step with stronger generators plus explicit verification; [[gorilla]] targets API-name and argument correctness over documentation.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2302.04761 (arXiv v1, 9 Feb 2023)
- Corrections to the previous card version:
  - "Roberto Dessi" → "Roberto Dessì"; author list trimmed to the first six plus "et al." per the card standard (full list adds Nicola Cancedda and Thomas Scialom).
  - "a subset of CCNet, and five text-form APIs" → kept, with the specific tool implementations added (Atlas-large/Atlas-xxl, BM25 over the KILT Wikipedia dump, 600M NLLB, Python calculator) from §3 and App. A.1.
  - "the appendix states a default threshold of at least 5% probability" → correct, and the per-tool exception now recorded: τ_s = 0.0, k = 20, m = 10, τ_f = 0.5 for calculator and machine translation (App. A).
  - "the appendix gives a default threshold of 1.0" → correct; Table 2 reports yields at τ_f ∈ {0.5, 1.0, 2.0} but no downstream accuracy per threshold, so the card no longer implies 1.0 was selected by measured benefit.
  - "a 6.7B model with tool access can beat much larger pure-LM baselines" → replaced by the specific tables: it holds on LAMA (Table 3) and the math benchmarks (Table 4), and does not hold on WebQS / NQ / TriviaQA against GPT-3 175B (Table 5).
  - "translation candidates containing a non-English chunk inside surrounding English context" → correct, and the exact rule added: 10-token chunk, fastText confidence above 0.8, numeric/symbol-only chunks dropped (App. A.1).
  - Year line now gives the arXiv v1 month; added missing **Source type** field and a Recipe ledger.
- Removed as unsupported by the source:
  - "while preserving general language-modeling ability" as an unqualified claim — replaced with the Table 8 perplexities and the footnote-8 caveat that perplexity with API calls enabled was not measured.
  - "Establishes the precursor pattern for later tool-data synthesis" and "In modern terms, it is the annotation bootstrap stage before later systems move to stronger teachers" — claims about later work, not results of this paper; the lineage is kept only as wikilinks under Connections.
  - "Short, local, high-information tool outputs are easiest to learn from with self-supervision" and the other "Practical lessons for modern pipelines" bullets — the paper reports no comparison across tool-output lengths or information density.
  - "Table 2 ... useful for seeing how much the filter compresses raw candidates" — Table 2 gives accepted counts only; the number of raw candidates before filtering is not reported.
- Not reported by the source: size of C before annotation; epochs, optimizer, sequence length, and compute for the fine-tune; number of raw candidate calls before filtering; per-threshold downstream accuracy.
