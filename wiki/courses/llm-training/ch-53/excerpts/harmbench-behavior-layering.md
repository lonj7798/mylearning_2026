---
chapter: ch-53
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/harmbench-data.md
source_url: https://arxiv.org/abs/2402.04249
primary_text_checked: arXiv:2402.04249 (HarmBench, ICML 2024), 2026-09-17
revised: 2026-09 (generality revision)
---

# Excerpt: measuring the policy's safety, not a classifier's accuracy

Used by ch-53 §8. The quantity measured is whether the checkpoint under test produces the harmful
behaviour when asked, scored by a separate released classifier. Feeding a moderation test set to the
checkpoint and grading its labels measures something else — the checkpoint's skill as a moderator.

## Three separated layers (§4.1–4.3)

1. **Behaviour.** A manually authored request, tagged with a semantic category (7 of them: cybercrime,
   chemical/biological, copyright, misinformation, harassment, illegal activities, general harm) and a
   functional category (standard, copyright, contextual, multimodal). Behaviours were written from a
   distilled summary of the acceptable-use policies of OpenAI, Anthropic, Meta, and Inflection, then
   filtered to remove dual-intent requests that many benign users would also make.
2. **Attack.** A red-teaming method turns a behaviour into test cases: direct request, human jailbreak
   templates, token-optimization attacks (GCG and variants, PEZ, GBDA, UAT, AutoPrompt), attacker-LLM
   search (PAIR, TAP, zero-shot, stochastic few-shot), and evolution or persuasion methods (AutoDAN,
   PAP).
3. **Scoring.** The target model generates T tokens with greedy decoding; a classifier decides whether
   the completion exhibits the behaviour or is a clear attempt at it. Incorrect content still counts as
   success when the attempt is genuine — the standard exists to separate safety failure from
   capability failure (§4.3, App. B.1).

ASR for a method g on model f over behaviours is `ASR(y, g, f) = (1/N) Σ_i c(f_T(x_i), y)` with
`c = 1` for a successful test case (§3.1).

## The two numbers that pin the measurement

**Generation length.** "The number of tokens generated during evaluation can have a drastic effect on
ASR"; Figure 2 shows the choice changing ASR by up to 30%, and the paper standardizes it at
**N = 512** so the metric converges (§3.2). Cross-paper comparisons without this parameter are not
comparable.

**Scorer agreement with human judgments** (Table 3, manually labelled validation set):

| Scorer | Standard behaviours | Contextual behaviours |
|---|---|---|
| HarmBench classifier (fine-tuned Llama-2-13B-Chat) | 94.53 | 90.5 |
| GPT-4 | 89.8 | 85.5 |
| GPTFuzz (fine-tuned RoBERTa) | 77.36 | 71.5 |
| Llama-Guard | 68.41 | 64.0 |
| ChatGLM | 65.67 | 62.5 |
| AdvBench substring matching | 71.14 | 67.5 |

Substring matching is not prohibited by the paper; it is about 23 points less accurate than the
released classifier on standard behaviours, and the paper's robustness tests show why: completions that
refuse first and then comply, benign paragraphs, and completions for unrelated harmful behaviours all
break scorers that look at prefixes (§3.2).

**Two classifiers, two roles** (App. B.2). The test classifier is fine-tuned from Llama-2-13B-Chat and
reaches 93.2% agreement with human labels (41 errors); a separate validation classifier fine-tuned
from Mistral-7B base on half the fine-tuning set reaches 88.6% (51 errors), and the two error sets
intersect in only 26 examples. The validation classifier is the one to use inside an optimization loop;
the paper does not allow direct optimization against the test metric. Copyright behaviours use a
hashing-based classifier with MinHash matching over overlapping chunks, because "attempting" to
reproduce copyrighted text is not the quantity of interest.

**Validation/test behaviour split** (§4.1). Attacks and defenses are developed on the validation
behaviours and judged on the test behaviours.

## The complement

HarmBench measures compliance with harmful requests. The same checkpoint must be measured for refusal
of safe requests, which HarmBench does not cover; ch-53 §8 uses [[xstest]] for that, and reports the
pair (ASR, over-refusal) together.

Related: [[harmbench-data]], [[xstest]], [[ch-52]], [[read]].
