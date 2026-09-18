<!-- scope: rephrase web documents into cleaner synthetic styles for compute- and data-efficient pretraining
     deps: [[fineweb]]
     see-also: [[phi-textbooks]], [[nemotron-4-synthetic]], [[model-collapse]], [[magpie]], [[physics-of-lm-3]], [[synthetic-data-scaling-laws]]
-->

# Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling
- **Core Insight:** Pre-training a 1.3B model on C4 plus Mistral-7B-Instruct rephrases of the same documents, sampled 1:1, speeds up pre-training by about 3x and improves average zero-shot accuracy across 13 tasks by more than 2% relative to C4 alone (Abstract).
- **Guideline:** When the pretraining corpus is noisy web text and the compute budget is fixed, rephrase chunks of at most 300 tokens with a frozen instruction-tuned model and train on real and synthetic text in a 1:1 ratio, because removing the real half lowers accuracy on specialized-knowledge tasks (§5, Table 4). When the target evaluation is question answering, use the Q/A style; when it is general perplexity on encyclopedic text, use the Medium (Wikipedia) style (§5.2).
- **Authors:** Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly
- **Year:** 2024 (arXiv v1 2024-01; ACL 2024, pp. 14044–14072)
- **URL:** https://aclanthology.org/2024.acl-long.757/ (arXiv preprint: https://arxiv.org/abs/2401.16380)
- **Source type:** paper
- **Relevant topics:** synthetic pretraining, paraphrase augmentation, style diversity, data efficiency, web rewriting

## Abstract
The paper proposes Web Rephrase Augmented Pre-training (WRAP). An off-the-shelf instruction-tuned
model is prompted to paraphrase web documents in specified styles, and a language model is pre-trained
jointly on the real and the synthetic text. On C4, WRAP speeds up pre-training by about 3x; at the same
pre-training compute budget it improves perplexity by more than 10% on average across subsets of the
Pile and improves zero-shot question-answer accuracy across 13 tasks by more than 2% (Abstract). The
paper then studies how the rephrasing style changes performance, and attributes the gains to two factors:
the synthetic text adds style diversity that matches downstream evaluation style, and it is of higher
quality than the web scrape (Abstract).

## Key Contributions
- Defines WRAP: rephrase web documents with a frozen instruction-tuned model into four styles, then
  pre-train on real and synthetic text sampled 1:1 (§3.1).
- Reports that synthetic data allows training equivalent models with 5x less data or 3x less compute (§1).
- Reports that a 350M model trained on 15% of C4 outperforms a 1.3B model trained on the entire C4 (§1).
- Measures style effects: Q/A-style rephrases give the largest zero-shot gains, and combining styles does
  not beat Q/A alone on zero-shot tasks (§5.2, Table 5, Table 6).
- Separates rephrasing from generic augmentation: synonym replacement and random deletion via
  NL-Augmenter do not reproduce the gain (§6, Figure 6).
- Checks that the gain is not information leakage from the rephrase model by comparing real-synthetic
  cosine similarity against several baselines (App. C, Figures 8–9).

## Key Figures/Tables to Study
- **Figure 1** — the WRAP pipeline, the zero-shot accuracy-vs-tokens curve labelled "~3x Faster", and
  average Pile perplexity across data-pool sizes.
- **Table 1 / Table 2** — 1.3B results on 8 general-understanding and 5 specialized-knowledge tasks
  (13 tasks total), against Half C4, Full C4, RefinedWeb, Pythia-Pile, and TinyLlama baselines.
- **Figure 6** — rephrasing versus generic augmentation (synonym replacement, random deletion).
- **Figure 7** — Pile perplexity by rephrasing style.
- **Figures 8–9** — cosine-similarity checks for semantic preservation and for MRPC-style rephrases.
- **Figures 10–11** — Flesch–Kincaid reading level, type-token-ratio diversity, and syntactic complexity
  (tree depth, mean dependency distance) of synthetic versus real text.

## Technical Details
- **Seed corpus:** C4 (§3.1). Each rephrased example is at most 300 tokens, chosen because prompting the
  model to rephrase more than 300 tokens often lost information (§3.1).
- **Rephrase model:** a frozen Mistral-7B instruction-tuned model (§3.1). Ablations also use Qwen-1.8B-chat,
  Mistral-7B-chat, and Vicuna-13B-chat-v1.3 (§6).
- **Styles:** Easy (understandable by a toddler), Medium (high-quality English as on Wikipedia), Hard
  (terse and abstruse), Q/A (conversational question-answering) (§3.1; prompts in App. G).
- **Mixing:** real and synthetic data sampled in a 1:1 ratio (§3.1).
- **Output cleanup:** App. B removes extraneous introductions such as "Here's a paraphrase...",
  "The following...", or the keyword "high-quality English", by splitting into sentences with the NLTK
  sentence splitter and dropping the segment before a "\n\n" or ":" delimiter; a paraphrase still carrying
  a flagged keyword is discarded entirely. Manual inspection puts the residual error rate below 0.1%
  (App. B.1).
- **Perplexity result:** averaged over multiple Pile subsets, WRAP models improve perplexity by 50% over
  models trained on real data alone; on ArXiv and HackerNews the reduction is close to 3x (§4).
- **Learning speed:** at the first checkpoint (10B tokens), WRAP average Pile perplexity is already below
  C4 training at 15 checkpoints, which the authors describe as a 15x pre-training speed-up (§4).
- **Zero-shot result:** Synthetic+C4 (85B real tokens) averages 49.4 on the 8 general-understanding tasks
  versus 47.3 for Full C4 (170B real tokens) (Table 1), and 45.5 versus 43.5 on the 5 specialized tasks
  (Table 2); WRAP rows are averaged over 3 runs (Table 1 caption).
- **Generation cost:** about 3M tokens per hour on a single A100 with Mistral-7B; Qwen-1.8B gives 3x higher
  token throughput at comparable downstream performance (§6).

## Recipe ledger
Moved to [[rephrasing-the-web-recipe]] to keep this card under the 120-line limit.

## Findings relevant to generality
- The paper frames rephrasing as adding style diversity that matches downstream evaluation style rather
  than adding information, and supports the "no new information" side with cosine-similarity checks
  against real-real, continuation, and MRPC baselines (App. C).
- Keeping real data in the mixture is presented as necessary so that the model still handles noisy
  user-facing text with typos and linguistic errors (§3.1); Table 4 measures the specialized-task cost of
  dropping it.
- Combining several synthetic styles does not improve zero-shot performance over Q/A style alone
  (Table 5, Table 6 captions), which limits "more style diversity is always better".
- Untested conditions: all trained models are 128M, 350M, or 1.3B parameters (§3.2). No result above 1.3B.

## Connections
- [[phi-textbooks]], [[hf-cosmopedia]] — generate synthetic pretraining text from topic seeds instead of
  rewriting existing documents.
- [[physics-of-lm-3]] — measures how diverse restatements affect knowledge extractability; its Result 2
  reports LLaMA2 rewrites keep capacity near 2 bit/param (§5.1 of that paper).
- [[synthetic-data-scaling-laws]] — studies how the WRAP-style gain scales with synthetic data volume.
- [[model-collapse]], [[strong-model-collapse]] — recursive-retraining failure modes; WRAP is not
  self-iterative, so the settings differ.
- [[kimi-k2]] — states that WRAP inspired its knowledge rephrasing prompts (§2.2 of that report).
- [[fineweb]], [[dolma]] — filtering the crawl, the alternative to rewriting it.
- [[self-instruct]], [[magpie]] — synthetic instruction data, a different training stage.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2401.16380 (v1, 2024-01-29) and
  https://aclanthology.org/2024.acl-long.757/ (title and author list only).
- Corrections to the previous card version:
  - "Richard Bai" → "He Bai" (arXiv v1 author block; the ACL Anthology metadata renders the name as
    "Richard Bai").
  - "Figure 6 — the leakage sanity check via cosine similarity against MRPC-style rephrases" → Figure 6 is
    the comparison against generic augmentation; the MRPC cosine-similarity check is Figure 9 (App. C).
  - "Figure 7 — readability and lexical-diversity differences" → Figure 7 is Pile perplexity by style;
    reading level and type-token ratio are Figure 10 (App. C).
  - "Figure 8 — syntactic complexity" → Figure 8 is the semantic-similarity comparison; syntactic complexity (tree depth, mean dependency distance) is Figure 11 (App. C).
  - "more than 50% average perplexity reduction ... in the final ACL version" → the 50% figure is in §4 of
    arXiv v1; the abstract's same-compute-budget figure is "more than 10%".
  - Added the missing **Source type** field required by the card standard.
- Removed as unsupported by the source: "cut pretraining compute/data needs by a large margin" (replaced
  with the reported 3x compute / 5x data figures); "the core recipe does not use a learned quality filter"
  (the paper does not make this comparison; App. B describes the only filtering step); "Established a
  practical synthetic-data recipe that later work extended into larger-scale synthetic pretraining
  systems" and "Direct ancestor of [[nemotron-4-synthetic]], [[phi-textbooks]]" (phi-textbooks predates
  WRAP; the paper claims no descendants); "Wikipedia-like rephrasing helps readability and general
  pretraining quality" as a contribution claim (Figure 10 reports that Medium style raises reading level
  from 7–8 to 10, which is not the same claim).
- Not reported by the source: total GPU-hours for pretraining; optimizer, weight decay, and warmup
  schedule; the exact number of synthetic tokens generated per style.
