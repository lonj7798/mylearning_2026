<!-- scope: Greg Kamradt's Needle In A Haystack repository (Nov 2023) — single-needle long-context retrieval harness, its defaults, the original GPT-4-128K and Claude 2.1 runs, later multi-needle and v2 changes, and known validity problems
     deps: []
     see-also: [[ruler]], [[babilong]], [[longbench]]
-->

# Needle In A Haystack - Pressure Testing LLMs
- **Core Insight:** The harness inserts one sentence (the "needle") into Paul Graham essays at a chosen token depth, trims the context to a chosen length, asks a question the needle answers, and has a GPT-4 judge score the reply from 1 to 10; in the released GPT-4-128K run, 207 of 225 (length, depth) cells scored 10 (original_results at commit 7ffcec1).
- **Guideline:** When NIAH scores are compared across models or releases, fix and report the needle, question, prompt template (including any assistant prefill), judge, and length/depth grid, because the harness scores with an LLM judge (`evaluate_response`) and its Anthropic prompt was changed after the original Claude 2.1 run (commit 6ee3bf0, 2023-12-06).
- **Authors:** Greg Kamradt (repository owner); multi-needle support contributed by Lance Martin (commit c11e0da, 2024-03-06)
- **Year:** 2023 (repository created 2023-11-11; GPT-4-128K run 2023-11-08; Claude 2.1 run 2023-11-21)
- **URL:** https://github.com/gkamradt/LLMTest_NeedleInAHaystack (now redirects to github.com/gkamradt/needle-in-a-haystack); original code pinned at commit 7ffcec17ea09aa882ea2f0967d026ed6b8c30382
- **Source type:** released config/code
- **Relevant topics:** long-context evaluation, needle-in-a-haystack, retrieval, LLM-as-judge scoring, evaluation validity

## Summary
The repository implements a three-step test: place a fact or statement in a long context, ask the model to retrieve it, and repeat over context lengths and needle depths (README "The Test"). Results are plotted as a length × depth pivot table (README "Results Visualization"). The README links the two original analyses (OpenAI GPT-4-128K and Anthropic Claude 2.1) as tweet threads and a video, and stores their raw results in `original_results/`. In March 2024 the code was packaged as `needlehaystack` with multi-needle insertion; in May 2026 it was rewritten as "v2" with a `niah` CLI.

## Key Contributions
- A parameterized harness: needle text, haystack directory, retrieval question, length grid, depth grid, and linear or sigmoid depth spacing (README "The key parameters").
- Released raw results for the GPT-4-128K and Claude 2.1 runs (`original_results/`).
- The length × depth heatmap format used to report the result.
- Later additions: multi-needle insertion with even spacing (2024-03), and v2 task types with exact-match or fractional scoring (2026-05).

## Key Figures/Tables to Study
- `img/GPT_4_testing.png` and `img/Claude_2_1_testing.png`: the two original heatmaps.
- `LLMNeedleHaystackTester.py` at commit 7ffcec1: `insert_needle`, `generate_prompt`, `evaluate_response`.
- `Anthropic_prompt.txt` at commit 7ffcec1: the modified Anthropic prompt with an assistant prefill.

## Technical Details
All line numbers refer to `LLMNeedleHaystackTester.py` at commit 7ffcec1.
- **Needle (default):** "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day." (L24).
- **Question (default):** "What is the best thing to do in San Francisco?" (L26). Anthropic's Dec 2023 post quotes the question used for Claude 2.1 as "What is the most fun thing to do in San Francisco?".
- **Haystack:** all `.txt` files in `PaulGrahamEssays/`, concatenated repeatedly until the longest requested length is reached (L25, `read_context_files`).
- **Length grid (default):** 35 values from 1,000 to 200,000 tokens, linearly spaced and rounded (L28-30, L92).
- **Depth grid (default):** 35 values from 0% to 100%, linear, or sigmoid via a logistic with L = 100, x0 = 50, k = 0.1, which places more depths near 0% and 100% (L32-36, L100-103, L144-150).
- **Insertion:** 200 tokens are reserved for the system message, question, and response (L44, L328); the context is cut so context + needle fit; the needle goes at `int(len(context) × depth/100)`, moved back to the nearest preceding "." token so it starts at a sentence boundary; depth 100 appends it at the end (L323-358).
- **Tokenizers:** tiktoken for the tested OpenAI model; the Anthropic client tokenizer for Anthropic models (L137, L140).
- **Model call:** temperature 0, maximum 300 output tokens (L205-222). OpenAI prompt: system "You are a helpful AI bot that answers questions for a user. Keep your response short and direct", then the context, then the question followed by "Don't give information outside the document or repeat your findings" (L169-190).
- **Anthropic prompt file:** the same instruction in Human/Assistant format, with the assistant turn prefilled by "Here is the most relevant sentence in the context:" (`Anthropic_prompt.txt`, added in commit 6ee3bf0 on 2023-12-06).
- **Scoring:** LangChain `labeled_score_string` with GPT-4 at temperature 0, the needle as reference, and a rubric of 1 (unrelated), 3, 5, 7, 10 (completely accurate) (L143, L360-390). Each cell therefore costs one call to the tested model and one GPT-4 judge call.
- **Multi-needle (commit c11e0da, README at 8af85a8):** the first needle goes at `depth_percent`; the rest are spaced by `(100 − depth_percent) / N`, e.g. 10 needles from 40% land at 40, 46, …, 94.
- **v2 (commit f68fb20, 2026-05-29; README at HEAD):** task types `single` (one fact, exact-match scored), `multi` (N facts, fractional score), `uuid`, and `uuid_chain` (multi-hop links); JSONL rows store a recipe from which the exact context can be rebuilt. The README states v2 fixes a v1 multi-needle bug in which each needle's reported depth was shifted by the tokens of earlier needles.

## Findings relevant to long context
Counts below were computed on 2026-09-14 from the released result files at commit 7ffcec1.
- **GPT-4-128K (run 2023-11-08):** 15 lengths from 1,000 to 128,000 tokens × 15 depths from 0% to 100% = 225 cells; scores: 10 in 207 cells, 7 in 1, 5 in 1, 3 in 13, 1 in 3 (`OpenAI_Original_Results/OpenAI_OriginalResults.json`).
- **Claude 2.1 (run 2023-11-21):** 35 lengths from 1,000 to 200,000 tokens × 35 sigmoid-spaced depths; 1,225 files marked version 1, of which 673 scored 10; 151 further files are marked version 2, which the README does not explain (`Anthropic_Original_Results/`). Low-scoring responses include refusals such as "Unfortunately, the context does not mention anything about fun things to do in San Francisco" (200K tokens, 50% depth, score 1).

## Connections
- [[ruler]] — synthetic suite that extends single-needle retrieval with multi-key, multi-value, and multi-query needles plus tracing, aggregation, and QA tasks.
- [[babilong]] — bAbI reasoning tasks whose facts are spread through long distractor text, testing reasoning over retrieved facts rather than one retrieval.
- [[longbench]] — natural long-document tasks, used as a contrast to synthetic retrieval.
- [[longalign]], [[prolong]], [[qwen-long-context-synth]] — long-context training recipes; whether they use NIAH-style training data must be checked in those sources, not inferred from this repository.
- Related sources without a library card at the time of writing (cited with their own loci):
  - Anthropic, "Long context prompting for Claude 2.1" (2023-12-06): prefilling "Here is the most relevant sentence in the context:" raised Claude 2.1's score on this evaluation from 27% to 98%; Anthropic attributes the refusals to training that discourages answering from an out-of-place sentence (Interpretation).
  - Anthropic, "Introducing the next generation of Claude" (2024-03-04): a variant with one of 30 needle/question pairs per prompt and a crowdsourced corpus; Claude 3 Opus exceeded 99% recall and in some cases stated that the needle appeared to be artificially inserted.
  - Magic, "100M Token Context Windows" (2024-08-29): a needle stands out semantically from its haystack, so a model can detect it without using the rest of the context; this motivated the HashHop eval.
  - NoLiMa, arXiv:2502.05167 (abstract, Table 3): with minimal lexical overlap between question and needle, GPT-4o falls from a base score of 99.3 to 69.7 at 32K, and 11 of 13 models drop below 50% of their short-context baselines at 32K.
  - HELMET, arXiv:2410.02694 (§3.1, Figure 3): across 35 instruction-tuned models at 128K input length, the original NIAH has Spearman correlation ≤ 0.8 with every real-world task category.
  - Gemini API long-context documentation (last updated 2026-06-22, "Long context limitations"): NIAH evals are the most basic single-needle setup, and with multiple needles "the model does not perform with the same accuracy".
  - From Artificial Needles to Real Haystacks, arXiv:2406.19292 (Table 2): fine-tuning Mistral 7B on a needle-in-a-haystack dataset changed TriviaQA by −6.33 and NQ-Open by −6.73 points, while the authors' synthetic key-value data changed them by +0.11 and +0.37.

## Verification
- Checked on 2026-09-14 against: https://github.com/gkamradt/LLMTest_NeedleInAHaystack at commit 7ffcec17ea09aa882ea2f0967d026ed6b8c30382 (2023-12-06; README.md, LLMNeedleHaystackTester.py, Anthropic_prompt.txt, original_results/), README at commit 8af85a8fee (2024-03-25), and README at HEAD (v2, 2026-05); plus the related sources listed above at the stated loci.
- Corrections to the previous card version: needle "eat a sandwich at Dolores Park on a sunny day" → "eat a sandwich and sit in Dolores Park on a sunny day" (L24); "Evaluation metric: exact-substring match (simple) or LLM-judge" → the original harness scores with a GPT-4 judge on a 1-10 rubric (L360-390), exact-match scoring exists only in the v2 `single` task; "Context lengths tested: 1K, 4K, 8K, 16K, 32K, 64K, 128K" → GPT-4 run 15 linearly spaced lengths 1,000-128,000, Claude 2.1 run 35 lengths 1,000-200,000; "depths 0%, 10%, 20%, …, 100%" → 15 depths (GPT-4 run), 35 sigmoid-spaced depths (Claude 2.1 run), 35 linear by default; multi-needle "(2024) … Anthropic multi-needle: 1–8 needles" → multi-needle insertion was added to this repository by Lance Martin on 2024-03-06, and multi-key / multi-value / multi-query are RULER task types; "Nov 2023 blog + GitHub" → GitHub repository plus linked tweet threads and video; "Data recipe that uses NIAH as training signal: longalign, prolong, qwen-long-context-synth" → not supported by this source; direct evidence on NIAH-style fine-tuning is arXiv:2406.19292 Table 2.
- Removed as unsupported by the source: "Paul Graham essays … mostly absent from post-2021 training cutoffs"; "many models have PG essays in training"; "Single NIAH: saturated for 2025 models — most score >95%"; "8-needle: Claude-3.5 ~90%, Llama-3.1-70B ~70%, weaker open models <50%"; "can pass NIAH at 128K while failing multi-hop reasoning at 32K" (replaced by the NoLiMa and HELMET results); "NIAH exposes RoPE-extension failure modes … non-uniform RoPE issues"; "Used by every 2024+ long-context release — Claude, GPT-4-128K, Llama 3, Qwen 2.5-1M, Gemini"; "Token-range 1K → 1M+"; "Cost: near-zero" (each cell needs a GPT-4 judge call); "became viral / de facto community standard".
- Not reported by the source: judge agreement with humans, variance across seeds or needles, results for models other than GPT-4-128K and Claude 2.1.
