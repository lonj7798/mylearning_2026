<!-- scope: the capability-specific synthetic-data pipelines of Llama 3 post-training (arXiv:2407.21783 §4.3)
     deps: [[llama-3]], [[dpo]]
     see-also: [[llama-3-recipe]], [[long-context-llama3]], [[rejection-sampling-finetuning]], [[tulu-3]], [[magpie]]
-->

# The Llama 3 Herd of Models — §4.3 Capabilities (synthetic-data pipelines)
- **Core Insight:** Llama 3 post-training builds a separate synthetic-data pipeline per capability, each with its own verifier: code uses execution feedback and produces over 2.7M synthetic SFT examples, long context uses three model-generated task families mixed at 0.1% of the SFT data, and factuality uses a knowledge probe that converts confidently wrong answers into refusals (§4.3.1, §4.3.4, §4.3.6).
- **Guideline:** When generating synthetic SFT data for a capability, pair generation with a checker that is cheaper than the generator and independent of it — a parser, a unit test, a language-ID filter, a reward model, or the source document — because the pipelines that report the largest effects in §4.3 are the ones where such a signal exists. Otherwise, when no such signal exists, the paper falls back on model-as-judge filtering with a binary correctness-and-style score and on human annotation (§4.3.1, §4.3.5).
- **Authors:** Llama Team, AI @ Meta
- **Year:** 2024 (arXiv v1 2024-07; v3 2024-11-23)
- **URL:** https://arxiv.org/abs/2407.21783
- **Source type:** official technical report
- **Relevant topics:** synthetic SFT data, execution feedback, backtranslation, rejection sampling, long-context synthesis, tool-use data, factuality and refusals

> This card is the library's owner for §4.3 of the Llama 3 report, read as a set of synthetic-data
> pipelines. [[llama-3]] is the owner card for the report as a whole; [[long-context-llama3]] is the
> owner card for the long-context material in §3.4.2 and §4.3.4, and is the card to cite for that topic.

## Summary
§4.3 of the Llama 3 report describes capability-specific work layered on top of the general post-training loop of SFT, rejection sampling, and DPO. Seven capabilities are covered: code (§4.3.1), multilinguality (§4.3.2), math and reasoning (§4.3.3), long context (§4.3.4), tool use (§4.3.5), factuality (§4.3.6), and steerability (§4.3.7). Two of them start from a domain expert model created by branching the pre-training run and continuing on a domain-heavy mixture; the expert then collects annotations and performs rejection sampling for its domain. The report states that training Llama 3 405B on its own generated data was not helpful and could degrade performance, and that execution feedback was introduced to supply an external source of truth (§4.3.1).

## Key Contributions
- Domain experts produced by branching pre-training: a code expert continued on 1T tokens of a mix that is more than 85% code, and a multilingual expert continued on a mix that is 90% multilingual tokens (§4.3.1, §4.3.2).
- Three synthetic code-data routes — execution feedback, programming-language translation, and backtranslation — with reported volumes (§4.3.1).
- Three long-context generators and the 0.1% mixing ratio that the ablations selected (§4.3.4).
- A knowledge-probing procedure that turns consistently informative-but-incorrect answers into trained refusals (§4.3.6).
- A statement of what is *not* synthetic: tool use relies on message-level human preference annotation and skips rejection sampling (§4.3.5).

## Key Figures/Tables to Study
- Figure 8 (§4.3.1) — a Python-to-PHP translation example used to augment low-resource language coverage.
- Figure 9 (§4.3.1) — generated code with and without the code-specific rejection-sampling system prompt.
- Figure 10 (§4.3.5) — a multi-step tool-use trajectory.
- Figure 11 (§4.3.6) — file-upload analysis, the file-type coverage described in §4.3.5.

## Technical Details
- **Code expert:** branched from the main pre-training run and continued on a 1T-token mix of mostly (>85%) code data, following a recipe similar to CodeLlama; the last several thousand steps are long-context finetuning to 16K on repo-level code; the expert is then aligned with code-targeted SFT and DPO and used for rejection sampling on coding prompts (§4.3.1).
- **Code synthetic totals:** over 2.7M synthetic examples used during SFT, of which approximately 1M are execution-feedback coding dialogues and approximately 1.2M are backtranslation dialogs for explanation, generation, documentation, and debugging (§4.3.1).
- **Execution-feedback loop:** problem descriptions generated from sampled real code snippets; solutions generated with good-programming rules in the prompt; correctness approximated by a parser and linter plus model-generated unit tests run in a container; failures are returned to the model with stdout, stderr, and return code for self-correction; only dialogs passing all checks are kept. About 20% of solutions were initially incorrect and then self-corrected (§4.3.1).
- **Code quality filter:** earlier Llama 3 versions act as a model-as-judge assigning binary 0/1 scores for code correctness and code style; only samples scoring 2 are retained. This filter initially caused a downstream regression because it removed challenging prompts disproportionately, so responses to the hardest coding data were revised until they met the criteria (§4.3.1).
- **Multilingual SFT mixture:** 2.4% human annotations, 44.2% data from other NLP tasks rewritten into dialog, 18.8% rejection-sampled data, 34.6% translated reasoning data (§4.3.2).
- **Multilingual rejection sampling:** generation temperature sampled from 0.2–1.0 in early post-training rounds and fixed at 0.6 in the final round; a language-match check between prompt and response runs before reward-model selection (§4.3.2).
- **Translation policy:** machine-translated data is avoided to prevent translationese and name, gender, and cultural bias; the single exception is synthetic quantitative-reasoning data, which is translated and which the report says produced strong MGSM gains (§4.3.2).
- **Math and reasoning:** pre-training math text converted to question-answer form; step-by-step solutions generated by Llama 3 and filtered on the correct final answer plus Llama 3 self-verification of the steps; outcome and stepwise reward models trained to remove traces with wrong intermediate steps; Monte Carlo Tree Search with the learned stepwise reward models used for harder prompts; interleaved text-and-Python reasoning filtered by code execution; incorrect generations reused by prompting Llama 3 to correct them (§4.3.3).
- **Long context, three generators (§4.3.4):** (1) question answering — long documents from the pre-training mix split into 8K chunks, an earlier Llama 3 prompted to write QA pairs conditioned on random chunks, with the full document used as context at training time; (2) summarization — hierarchical, summarizing 8K chunks with the strongest 8K-context model and then summarizing the summaries, plus QA pairs over the summaries requiring whole-document understanding; (3) long-context code reasoning — Python import graphs are parsed, files depended on by at least five other files are removed, and the model must identify the dependents and regenerate the missing code. Samples are bucketed by sequence length into 16K, 32K, 64K, and 128K.
- **Long-context mixing ratio:** ablations selected mixing 0.1% synthetically generated long-context data with the original short-context data as optimal across both short- and long-context benchmarks (§4.3.4).
- **Long-context DPO:** short-context-only DPO did not hurt long-context performance given a strong long-context SFT checkpoint, which the report attributes to DPO using fewer optimizer steps than SFT; the standard short-context DPO recipe was kept (§4.3.4).
- **Tool use:** core tools are Brave Search, a Python interpreter, and the Wolfram Alpha API. Data comes from message-level human preference annotation, and rejection sampling is not used because it gave no gain on the tool benchmarks. Synthetic data from earlier checkpoints bootstraps basic tool use so that annotators edit less. About 30% of the single-step tool dataset is filtered out for non-executable calls or formatting problems. Zero-shot function-calling data is grounded on function calls and definitions mined from The Stack, with Llama 3 writing the matching natural-language query; multi-turn function-calling data is produced by several differently-prompted Llama 3 agents generating domains, APIs, queries, calls, and responses (§4.3.5).
- **Factuality:** a six-step knowledge probe — extract a pre-training snippet, prompt Llama 3 for a factual question about it, sample answers, score correctness with the snippet as reference and Llama 3 as judge, score informativeness with Llama 3 as judge, and generate a refusal for answers that are consistently informative and incorrect. A limited set of human-labeled factuality data covers sensitive topics (§4.3.6).
- **Steerability:** system-prompt preference samples collected by having annotators design system prompts and then hold multi-turn conversations to test consistency; no synthetic generation is described (§4.3.7).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 3 code expert | 405B branch | mid-train | continued-pretraining tokens / mixture | 1T tokens, >85% code | arXiv:2407.21783v3 §4.3.1 | verified (2026-09-18) | cites Gururangan et al. 2020 and the CodeLlama recipe; no ablation reported |
| Llama 3 code expert | 405B branch | long-context | LCFT context length | 16K, last several thousand steps | §4.3.1 | verified (2026-09-18) | no ablation reported |
| Llama 3 multilingual expert | 405B branch | mid-train | continued-pretraining mixture | 90% multilingual tokens | §4.3.2 | verified (2026-09-18) | no ablation reported |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | synthetic code examples | >2.7M total; ~1M execution-feedback; ~1.2M backtranslation | §4.3.1 | verified (2026-09-18) | no per-route ablation reported |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | multilingual mixture shares | 2.4% human / 44.2% other NLP tasks / 18.8% rejection-sampled / 34.6% translated reasoning | §4.3.2 | verified (2026-09-18) | no ablation reported |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | long-context synthetic share of SFT mix | 0.1% | §4.3.4 | verified (2026-09-18) | §4.3.4: "through careful ablations", optimizing short- and long-context benchmarks jointly; the ablation itself is not tabulated |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | long-context sample length buckets | 16K, 32K, 64K, 128K | §4.3.4 | verified (2026-09-18) | no ablation reported |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | multilingual rejection-sampling temperature | 0.2–1.0 early rounds; 0.6 final round | §4.3.2 | verified (2026-09-18) | high temperature raised code-switching (§4.3.2) |
| Llama 3 (all sizes) | 8B/70B/405B | SFT | tool-use single-step data filtered out | about 30% | §4.3.5 | verified (2026-09-18) | non-executable calls and formatting issues |
| Llama 3 (all sizes) | — | preference | long-context DPO data | short-context only | §4.3.4 | verified (2026-09-18) | §4.3.4: short-context-only DPO did not degrade long-context results given a strong long-context SFT checkpoint |
| — | — | SFT | per-capability token counts, epochs, learning rates | — | — | not reported in §4.3 (general post-training hyperparameters are in §4.1–§4.2; see [[llama-3-recipe]]) | — |

## Findings relevant to distillation and self-training
- Self-generated data does not help the largest model: the report states that the 8B and 70B models improve when trained on data from a larger, more competent model, but that training the 405B on its own generated data was not helpful and could degrade performance; execution feedback was added as an external source of truth (§4.3.1).
- Iteration is explicit: code finetuning runs over multiple rounds, each round's improved model generating the next round's data (§4.3.1).

## Findings relevant to negative feedback
- Failures are used as content, not as gradient. Failed code solutions re-enter the prompt together with parser, linter, and test output for self-correction, and only passing dialogs are trained on (§4.3.1). Incorrect math generations are corrected by Llama 3 and the corrected version is trained on (§4.3.3).
- Consistently informative-but-incorrect answers become refusal targets — a negative outcome converted into a positive training target (§4.3.6).
- Wrong intermediate steps are removed by outcome and stepwise reward models rather than penalized (§4.3.3).

## Findings relevant to long context
- Applying the short-context SFT recipe alone caused significant regressions in the long-context capability obtained during pre-training, which is the stated reason for adding long-context SFT data at all (§4.3.4).
- Human annotation of long-context examples is described as largely impractical, so the long-context SFT data is predominantly synthetic (§4.3.4).
- Full detail and the pre-training side (§3.4.2) belong to [[long-context-llama3]].

## Findings relevant to agentic training
- Tool use is the one capability where the report says rejection sampling was not used, because it gave no observed gain on the tool benchmarks; annotation is at the message level so that each tool call and each post-tool reasoning step gets feedback, and annotators may not rank or edit tool outputs (§4.3.5).

## Connections
- [[llama-3]] — owner card for the report as a whole; this card expands its §4.3.
- [[llama-3-recipe]] — the general post-training hyperparameters that §4.3 layers on top of.
- [[long-context-llama3]] — owner card for §3.4.2 and §4.3.4.
- [[rejection-sampling-finetuning]] — the method §4.2.2 and §4.3 rely on outside tool use.
- [[tulu-3]], [[hf-alignment-handbook]] — open recipes with the same staged structure and published mixtures.
- [[magpie]], [[open-thoughts]], [[hf-cosmopedia]] — synthetic-data pipelines with fully released corpora, in contrast to this report, which describes the pipelines without releasing them.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2407.21783 (arXiv v3, 23 Nov 2024), §4.3.1–§4.3.7. The previously cited blog https://ai.meta.com/blog/meta-llama-3/ (18 Apr 2024) was also re-read in full.
- Corrections to the previous card version:
  - The card mixed three artifacts — the April 2024 Llama 3 blog (URL and most quotes), the Llama 3 model card, and the `synthetic-data-kit` repository — while its slug and title claim the synthetic pipeline. It is now an extract of the artifact that actually documents that pipeline, §4.3 of the Llama 3 report.
  - "the blog says synthetic data is used heavily across coding, math, multilinguality, reasoning, long context, tool use, and factuality" → the April 2024 blog does not contain the word "synthetic" or any synthetic-data description; those capabilities are enumerated in §4.3 of the report.
  - "the exact per-domain mixture is not disclosed" and "the exact synthetic-data mix … not public" → §4.3.2 gives the multilingual SFT shares and §4.3.4 gives the 0.1% long-context share; §4.3.1 gives the code volumes.
  - "Meta describes a pipeline that asks Llama 3 to generate factual questions from pretraining snippets …" attributed to the blog → §4.3.6 of the report.
  - "DPO uses the most recent preference data from the strongest recent round" → §4.3.4 states only that the standard short-context DPO recipe is kept over long-context SFT checkpoints; the round structure belongs to §4.1–§4.2 and to [[llama-3-recipe]].
  - "Preference noise: if ranking quality drops, the whole loop degrades quickly" and "synthetic data can amplify style quirks if QA is weak" → not statements of the source.
- Removed as unsupported by the source: the `synthetic-data-kit` 4-step CLI flow (`ingest`, `create`, `curate`, `save-as`), which is a separate repository and not part of the Llama 3 report or blog; "no separate synthetic-preference corpus is publicly described"; "keep the prompt pool tight"; the "Risks + Gotchas" list; "15T tokens" and "four times more code" (pre-training facts belonging to [[llama-3]]); the claim that the model card summarizes the tuned versions as SFT + RLHF, which was not re-verified against a model-card artifact.
- Not reported by §4.3: per-capability token counts, epochs, and learning rates; the prompt templates used by any generator; the size of the human-annotated tool-use and factuality sets; contamination checks for the synthetic corpora; whether any of these corpora were released.
