---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: Kim et al. — "SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization" (library card [[soda]] is not verified; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2212.10465
primary_version: arXiv:2212.10465v3 (v1 2022-12; v3 2023-10)
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary source for the 2026-09 revision of read.md)"
---

# Excerpt: SODA — CO3 framework, filters, and COSMO evaluation

Authors: Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, et al. (Allen Institute for AI; Seoul National University; University of Washington; USC; University of Pittsburgh; CMU). Checked against the v3 PDF on 2026-09-15. Used by ch-25 `read.md` §1, §5, and the Recipe.

## Differences from the library card [[soda]]
The card states Atomic10x has "10M" triples (not stated in the paper), four relations "xWant, xNeed, xEffect, xReact" (the paper uses six x-relations), a "1–2 sentence narrative" (paper: two or three sentences), "4–10-turn" dialogues and "avg 20 tokens per turn" (paper: filter keeps 4–20 turns; average utterance length 16.1, unit not stated), "Cost ~$10K" (paper: about $0.02 per triple-to-dialogue), COSMO transfer to "EmpatheticDialogues" (paper's out-of-domain human evaluation uses DailyDialog), and a "Self-Instruct → SODA → Prosocial-Dialog" lineage (ProsocialDialog, arXiv 2022-05, precedes SODA).

## Motivation (§1)
> "We find that sampling from an LLM without contexts results in dull conversations (§3.3). Because commonsense knowledge graphs cover a wide range of everyday situations (West et al., 2022), conditioning on them results in a broad spectrum of conversations. Moreover, since LLMs are prone to hallucinations (Weidinger et al., 2021), the seed commonsense knowledge can help them stay on a sensible generation path."

## CO3 steps (§2, App. A)
1. Retrieve social commonsense from Atomic10x: "we only retrieve triples related to social (rather than, e.g., physical) commonsense" (§2.2). Relations used: xIntent, xWant, xReact, xAttr, xNeed (App. A.1) and xEffect (Table 3 shares: xAttr 18%, xEffect 17%, xIntent 23%, xNeed 7%, xReact 25%, xWant 11%).
2. Triple to sentence with templates (Table 8), e.g. `xReact: [Head]. Now PersonX feels [Tail].`, `xNeed: PersonX [Tail in past tense]. [Head].`; person variables replaced with Top-1K common US SSN names (§2.3).
3. Sentence to narrative: `[sentence-form commonsense] Rewrite this story with more specific details in two or three sentences:` with text-davinci-002, temperature 0.9, top-p 0.95, frequency penalty 1.0, presence penalty 0.6 (App. A.1). "We find long narratives tend to be driven far away from the original commonsense knowledge."
4. Infer the second speaker when the triple has one person; generate the conversation with the first speaker's name as an utterance prefix (§2.4).

Example (Table 1): Head "PersonX moves a step closer to the goal", Relation xNeed, Tail "to take the first step" → narrative "Madeleine took the first step towards her goal, and with her coach's encouraging words, she moves one step closer." → an 8-utterance Madeleine–Coach conversation.

## Post-processing (§3.1)
> "Starting with an initial set of 2.2 million conversations sampled from GPT-3.5, we: (1) use lexical pattern matching to filter out conversations with erroneous patterns – e.g., repetition and omission of speaker prefixes (6.3%); (2) remove conversations that have less than four turns or more than twenty turns (5.7%); (3) remove conversations with more than two speakers (11.3%); and (4) remove conversations where at least one of the speakers was identified as non-human (e.g., broomstick, imaginary friend, dog; 5.6%)."

- Safety: Canary "needs intervention" discarded (4.3%); Rewire API above 0.5 for violence, hate, or sexually explicit discarded (~1%).
- Commonsense check: 88% of 100 human-annotated pairs include the seed knowledge; GPT-3.5 zero-shot classifier (precision 97 for "yes") removes pairs lacking the head event.
- > "After all filtering, 68.9% of the initial conversations remain, which form the 1,486,896 conversations in SODA."

## Statistics (§3.2, Table 2)
| Dataset | #Dialog | Avg. #Turns | Avg. Utt. Length | Lexical Diversity (MTLD) |
|---|---|---|---|---|
| DailyDialog | 13K | 7.9 | 14.6 | 63.0 |
| PersonaChat | 11K | 14.8 | 14.2 | 43.6 |
| EmpatheticDialogue | 25K | 4.3 | 13.7 | 64.2 |
| BlendedSkillTalk | 7K | 11.2 | 13.6 | 64.2 |
| ProsocialDialog | 58K | 5.7 | 20.0 | 60.2 |
| SODA | 1.5M | 7.6 | 16.1 | 68.0 |

- > "It contains more than 11 million utterances ... In total, SODA consists of 300 million tokens"
- > "to go from a commonsense triple to a dialogue costs about $0.02, and 10 queries take less than 2 minutes, counting our full filtration pipeline." (§3.2)

## Contextualization ablation (§3.3)
> "we compare SODA with dialogues naively sampled from GPT-3.5 without any given context. We sample 100 dialogues using the same hyperparameters and the basic filtering steps in CO3, but with the following prompt: 'The following is a long in-depth conversation between two people.\nPerson 1:.'"

> "judges significantly prefer context-grounded conversations. Conversations sampled without context are not only less specific and less interesting, but also exhibit lower lexical diversity" (Fig. 3: all differences significant except Natural Flow; MTLD 68.0 vs 63.1)

## COSMO (§4–5)
> "The model is trained to generate a target response r when given n, i, and c – i.e., p(r|n, i, c)." (n narrative, i speaker instruction, c dialogue context) Training adds ProsocialDialog; "we randomly drop narrative n and role instruction i 30% and 50% of the time, respectively." Built on LM-adapted T5, 3B and 11B.

Table 5 (unseen DailyDialog, 100 test examples, three judges; Natural / Consistent / Specific / Overall):
| Comparison | Other model | COSMO-3B |
|---|---|---|
| vs BlenderBot-3B | 23 / 26 / 39 / 28% | 77 / 74 / 61 / 72% |
| vs GODEL-L | 13 / 14 / 15 / 14% | 87 / 86 / 85 / 86% |
| vs Koala-7B | 30 / 34 / 30 / 29% | 70 / 66 / 70 / 71% |
| vs Vicuna-7B | 42 / 42 / 44 / 42% | 58 / 58 / 56 / 58% |
| vs Ground Truth | 43 / 45 / 46 / 45% | 57 / 55 / 54 / 55% |
