<!-- scope: Conversation Chronicles (arXiv:2310.13420, UNIST, EMNLP 2023) — 1M-session / 200K-episode multi-session dialogue dataset generated with gpt-3.5-turbo-0301 from NLI-linked SODA event chains, 5 time-interval classes and 10 speaker relationships; ReBot (T5-base summarizer + BART-large generator, about 630M parameters)
     deps: [[soda]]
     see-also: [[msc-beyond-goldfish-memory]], [[locomo]], [[longmemeval]], [[realtalk]], [[baize]], [[ultrachat-construction]]
-->

# Conversation Chronicles: Towards Diverse Temporal and Relational Dynamics in Multi-Session Conversations
- **Core Insight:** Prompting ChatGPT (gpt-3.5-turbo-0301) with a 5-event chain, a time interval between sessions and one of 10 speaker relationships produced 200K episodes × 5 sessions (1M sessions); human raters scored 5K sampled episodes 4.33 overall out of 5 (Table 4) and rated 0.5K episodes above 0.5K MSC episodes on consistency (4.71 vs 3.87), coherence (4.51 vs 3.66) and time-interval fit (4.82 vs 4.05) (Figure 3).
- **Guideline:** When synthesizing multi-session dialogue with an LLM, condition each session on the earlier sessions' events and time gaps plus an explicit relationship, and filter speaker and format errors automatically, because this setup was rated above MSC on all compared criteria (§3.4, Figure 3); the paper does not test whether such data improves a general-purpose LLM, and its sessions are short (11.7 turns, 18.03 words per turn; Table 3).
- **Authors:** Jihyoung Jang, Minseong Boo, Hyounghun Kim (Artificial Intelligence Graduate School, UNIST)
- **Year:** 2023 (arXiv v1 2023-10; EMNLP 2023)
- **URL:** https://arxiv.org/abs/2310.13420
- **Source type:** paper
- **Relevant topics:** synthetic multi-turn dialogue, multi-session conversation, long-term memory, LLM data distillation, event graphs, NLI filtering, dialogue summarization

## Abstract
Open-domain chatbot research has focused on short single-session dialogue and has not addressed context from multiple earlier sessions, in particular the time intervals between sessions and the relationships between speakers. The paper introduces Conversation Chronicles, a 1M multi-session dialogue dataset generated with an LLM, in which time intervals and fine-grained speaker relationships are included. Human evaluation shows that episodes reflect these properties while staying coherent and consistent across sessions. The paper also proposes ReBot, a dialogue model with chronological summarization and dialogue generation modules using about 630M parameters, which shows long-term context understanding with high human engagement scores when trained on the dataset (Abstract).

## Key Contributions
- A multi-session dataset with time intervals from "a few hours" to "a couple of years" and 10 speaker relationships, which the authors describe as the first open-domain dialogue dataset with fine-grained speaker relationships (§1, §2).
- An event-graph construction that links SODA narratives by NLI entailment so that the 5 events of an episode are related (§3.1).
- ReBot: a T5-base chronological summarizer of past sessions and a BART-large response generator conditioned on relationship, time interval and summary (§4).
- Human evaluations of dataset quality, comparison with MSC, generated-episode quality and live chats against MSC 2.7B (§5-6).

## Key Figures/Tables to Study
- Figure 2 (collection process), Tables 1-3 (time-interval, relationship and scale statistics).
- Table 4 and Figure 3 (dataset quality; comparison with MSC), Table 5 and Figure 5 (ReBot quality; live chats vs MSC 2.7B).
- Appendix A (full prompts), Appendix B (filters), Appendix C (training settings), Table 24 (scores per relationship).

## Technical Details
- **Event pairing (§3.1):** each SODA narrative is one event. All event pairs are classified by a BERT-base model fine-tuned on SNLI as entailment, neutral or contradiction; only entailment pairs are kept.
- **Event graph (§3.1):** events are nodes in a directed graph (premise → hypothesis order, to prevent temporal contradiction). All event sequences of length 5 are extracted, and among sequences with more than 3 events in common only one is kept.
- **Time intervals (§3.2, Table 1):** one of "a few hours later", "a few days later", "a few weeks later", "a few months later", "a couple of years later" is picked at random for each consecutive session pair. Counts: 159,975 / 159,928 / 160,670 / 160,050 / 159,377 (sum 800,000 = 200K episodes × 4 gaps, derived). MSC has 3,497 (1-7 hours) and 3,510 (1-7 days) and no longer intervals (Table 1).
- **Relationships (§3.2, Table 2, App. A):** ChatGPT receives all events of an episode and a list of 10 relationships and selects the most appropriate one. Counts: Classmates 66,090 (33.05%), Neighbors 49,521 (24.76%), Co-workers 28,856 (14.43%), Mentee and Mentor 16,035 (8.02%), Husband and Wife 13,486 (6.74%), Patient and Doctor 6,980 (3.49%), Parent and Child 6,514 (3.26%), Student and Teacher 5,018 (2.51%), Employee and Boss 4,811 (2.41%), Athlete and Coach 2,689 (1.34%).
- **Session generation (§3.3, App. A):** one prompt per session contains the relationship, the previous session's event, the time interval, and today's event, and asks the two speakers to link to past topics if necessary. The model is "gpt-3.5-turbo-0301", fixed for reproducibility (App. A).
- **Scale (Table 3):** 1M sessions, 200K episodes, 11.7M turns, 11.7 turns per session, 18.03 words per turn. MSC train (up to 4 sessions): 4K sessions, 1K episodes, 53K turns, 13.3 turns per session. Derived from Table 3: about 211 words per session and about 1,055 words per 5-session episode.
- **Automatic filtering (§3.4, App. B):** remove every episode containing a session with more than two speakers, unclear utterance-speaker alignment, speakers outside the pre-defined relationship, or stage directions; remove harmful data with OpenAI Moderation. The number of removed episodes is not reported.
- **Dataset human evaluation (§3.4, §5.3, Table 4):** 5K episodes; Consistency 4.41 (std 0.80), Coherence 4.04 (1.06), Time interval 4.46 (0.77), Relationship 4.40 (1.08), Overall 4.33 (the text gives 4.34). Evaluators: 41 from a professional agency and 5 in-house (§5.2 footnote).
- **MSC comparison (§5.4, Figure 3):** 0.5K episodes from each dataset, each rated by three evaluators and averaged; MSC / Conversation Chronicles: Consistency 3.87 / 4.71, Coherence 3.66 / 4.51, Time interval 4.05 / 4.82, Overall 3.86 / 4.68. Relationship is excluded because MSC has none.
- **ReBot (§4, App. C):** summarizer T5-base (222M) trained on ChatGPT summaries of 100K sampled sessions (20K episodes), 80K for training and 20K for validation/test; generator BART-large (406M) models P(c | r, t, s, h) with input `<relationship> r <t_N> s_{N−1} <user> u_1 <bot> c_1 ...`, where c is the next utterance, r the relationship, t the time interval, s the summary and h the current dialogue context.
- **ReBot evaluations:** summary quality 4.3/5 on 3K summaries from sessions 2-4 (§5.5, §6). Generated episodes from 1K first sessions (0.1K per relationship), continued autoregressively: Engagingness 4.78, Humanness 4.74, Memorability 4.14, Overall 4.55 (Table 5). Live chats, 50 per model, at least 6 turns, in-house evaluators; MSC 2.7B / ReBot: Engagingness 3.24 / 4.06, Humanness 2.92 / 4.52, Memorability 2.62 / 4.12, Overall 2.93 / 4.23 (§5.6, Figure 5). Per-relationship engagingness ranges from 4.70 (Patient and Doctor) to 4.96 (Employee and Boss) (Table 24).
- **Ablation (§6, App. F, Tables 22-23):** without time-interval input the model produces generic time references; without relationship input it does not keep a consistent relationship. These are shown as qualitative examples only; no ablation scores are reported.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| ReBot summarization module (T5-base) | 222M | SFT | Training data | 80K ChatGPT-written session summaries (from 100K sampled sessions; 20K held for val/test) | arXiv:2310.13420v1 §4.1 | verified 2026-09-14 | no ablation reported |
| same | 222M | SFT | Optimizer; loss; schedule | AdamW; cross-entropy; linear scheduler | §5.1, App. C | verified 2026-09-14 | no ablation reported |
| same | 222M | SFT | Batch; max input / output length | 32; 512 / 128 (unit not stated) | App. C | verified 2026-09-14 | no ablation reported |
| same | 222M | SFT | Epochs; compute | max 5 epochs with early stopping; about 6 hours on 8 NVIDIA RTX A6000 | App. C | verified 2026-09-14 | — |
| ReBot generation module (BART-large) | 406M | SFT | Training data | 160K-episode train split (20K val, 20K test) | §5.1 | verified 2026-09-14 | no ablation reported |
| same | 406M | SFT | Optimizer; loss; schedule | AdamW; cross-entropy; linear scheduler | §5.1, App. C | verified 2026-09-14 | App. F: qualitative ablation of time and relationship inputs only |
| same | 406M | SFT | Batch; max input / output length | 16; 1024 / 128 (unit not stated) | App. C | verified 2026-09-14 | no ablation reported |
| same | 406M | SFT | Epochs; compute | max 3 epochs with early stopping; about 3 days on 8 NVIDIA RTX A6000 | App. C | verified 2026-09-14 | — |
| both modules | — | SFT | Peak LR, warmup, weight decay, early-stopping criterion | not reported (checked §5.1, App. C) | — | not reported | — |

## Findings relevant to generality, negative feedback, long context, distillation
- **Distillation data pipeline:** stage = dataset synthesis for dialogue-model SFT; prompt types = relationship selection, session generation, session summarization (App. A); teacher = gpt-3.5-turbo-0301, sampling parameters not reported; quality control = NLI entailment linking, rule filters, OpenAI Moderation, and human rating of a 5K sample (§3.1, §3.4).
- **Negatives:** event pairs labeled neutral or contradiction are discarded, and filtered sessions remove their whole episode (§3.1, App. B). This is discarding (negative marginal value); no negative training signal is used.
- **Long context:** ReBot replaces full history with per-session summaries and uses a 1024-length generator input (§4.1, App. C). Episodes are about 1,055 words (derived from Table 3), so the dataset does not test long-context inputs.
- **Generality:** evaluation is human rating of ReBot outputs and 50 live chats per model; no automatic benchmarks, no evaluation of LLMs trained on the data. The authors state that the limited set of time intervals and relationships may limit generalizability, and that a different generator LLM could produce different dialogues (Limitations).

## Connections
- [[soda]] — source of the event narratives (§3.1).
- [[msc-beyond-goldfish-memory]] — MSC, the multi-session dataset and 2.7B model used as comparison (§5.4, §5.6).
- [[baize]], [[prosocial-dialog]], [[ultrachat-construction]] — other LLM-distilled dialogue datasets; Baize and ProsocialDialog are cited as prior data distillation (§2).
- [[locomo]], [[longmemeval]], [[realtalk]], [[beam-10m-conversations]] — later long-term conversational memory benchmarks and datasets in this library.
- [[consistentchat]] — skeleton-guided multi-turn consistency synthesis, a later structure-conditioned approach.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2310.13420 (arXiv v1, 2023-10-20; only version; comments field: EMNLP 2023).
- Audit claims not found in the source: "POSTECH" → UNIST (title block). "ChatGPT checks each relationship fits the events" → ChatGPT selects the most appropriate of 10 relationships for the episode's events (§3.2, App. A). "drop redundant paths that share 3+ events" → sequences with more than 3 events in common (§3.1). "averages 4.34/5" → text 4.34, Table 4 Overall 4.33. "Ablations show both time and relationship information improve consistency" → ablation is qualitative examples only (App. F). "NLI-contradiction event pairs are discarded" → all non-entailment pairs (neutral and contradiction) are discarded (§3.1). "undefined relationships" filter → speakers not included in the pre-defined relationship (App. B). "each episode is only about 1K words" → not stated; derived from Table 3.
- Not reported by the source: generator sampling parameters, number of filtered episodes, learning rates, inter-annotator agreement.
