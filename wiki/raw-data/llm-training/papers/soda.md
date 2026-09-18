<!-- scope: multi-turn dialogue synthesis — SODA distils 1.5M single-session social dialogues from GPT-3.5 by contextualizing Atomic10x commonsense triples into narratives
     deps: [[self-instruct]]
     see-also: [[prosocial-dialog]], [[camel]], [[conversation-chronicles]], [[ultrachat-pipeline]], [[baize-construction]]
-->

# SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization
- **Core Insight:** Conditioning a teacher LLM on a social-commonsense triple expanded into a short narrative yields 1,486,896 dialogues (11M+ utterances, 300M tokens) that human judges prefer over both human-authored corpora and uncontextualized GPT-3.5 samples, with lexical diversity MTLD 68.0 vs 63.1 for the no-context samples (§3.1, §3.3, Table 2).
- **Guideline:** When synthesizing open-domain social dialogue, condition each generation on a distinct commonsense triple rendered as a narrative rather than prompting the teacher for conversation directly, because uncontextualized sampling from the same teacher was judged less specific, less interesting, and lexically less diverse (§3.3, Figure 3).
- **Authors:** Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, et al. (Allen Institute for AI, Seoul National University, University of Washington, USC, Pittsburgh, CMU)
- **Year:** 2022 (arXiv v1 2022-12; EMNLP 2023; text checked is arXiv v3, 2023-10-23)
- **URL:** https://arxiv.org/abs/2212.10465
- **Source type:** paper
- **Relevant topics:** synthetic dialogue data, commonsense grounding, knowledge-graph-seeded synthesis, dialogue distillation, single-session conversation

## Abstract
SODA is the first publicly available million-scale social dialogue dataset. It is produced by CO3, a framework that contextualizes social commonsense knowledge from a knowledge graph before distilling conversations from a large language model. CO3 retrieves social commonsense triples from Atomic10x, converts each triple to sentence form and then to a two-or-three-sentence narrative, infers the conversation participants, and prompts the teacher to generate a multi-turn conversation grounded in that narrative. Human evaluation finds SODA conversations more consistent, specific, and natural than prior human-authored datasets. Training the sequence-to-sequence model COSMO on SODA yields a conversation model that is more natural and consistent on unseen datasets than GODEL, BlenderBot-1, Koala, and Vicuna, and whose responses are sometimes preferred over the human-written gold responses in DailyDialog.

## Key Contributions
- **CO3 framework**: triple → sentence form → narrative → speakers → conversation, using social (not physical or event-centered) relations from Atomic10x (§2.2–§2.4).
- **SODA dataset**: 1,486,896 dialogues after filtering, from an initial 2.2M sampled conversations (68.9% retained) (§3.1).
- **COSMO-3B and COSMO-11B**, LM-adapted T5 models trained on SODA plus ProsocialDialog, evaluated by head-to-head human judgment (§4, §5).
- **Contextualization ablation**: SODA versus 100 dialogues sampled from the same teacher with no context, on six criteria plus interestingness (§3.3).

## Key Figures/Tables to Study
- **Table 1**: one full worked example from triple through narrative to conversation.
- **Table 2**: dataset statistics against DailyDialog, PersonaChat, WizardOfWikipedia, EmpatheticDialogue, BlendedSkillTalk, ProsocialDialog.
- **Figure 3**: head-to-head human preference, SODA versus uncontextualized GPT-3.5 samples.
- **Table 5**: COSMO-3B versus BlenderBot-3B, GODEL, Koala-7B, Vicuna-7B, and ground truth on DailyDialog.

## Technical Details
- **Teacher model:** GPT-3.5, specifically `text-davinci-002` (§2, footnote 2; Appendix A). Used for narrative generation, speaker inference, conversation generation, and as the zero-shot commonsense filter.
- **Seed knowledge:** Atomic10x (West et al., 2022); SODA is built on 1.5 million commonsense triples, restricted to x-relations (§3.2 "Diverse Content"; Appendix A). Relation shares in the released data: xIntent 23%, xAttr 18%, xEffect 17%, xNeed 7%, with xReact also present (Table 3).
- **Name handling:** person variables replaced with Top-1K US SSN applicant names (1990–2021) at contextualization time (§2.3); after generation, all names are re-randomized over the Top-10K list, covering 95% of applicants in that window (§3.1 "Name Bias Mitigation").
- **Filtering pipeline (§3.1):** lexical-pattern errors removed (6.3%); conversations with fewer than four or more than twenty turns removed (5.7%); conversations with more than two speakers removed (11.3%); conversations with a non-human speaker removed (5.6%); Canary safety model, discarding conversations needing intervention (4.3%); Rewire API toxicity, discarding above 0.5 on violence, hate, or sexually explicit (~1%); commonsense filter that drops narrative-conversation pairs lacking the head event, using GPT-3.5 as a zero-shot three-way classifier (95% of retained conversations contain the head event; a 100-pair human check found 88% instantiate the seed triple).
- **Dataset statistics (Table 2):** 1.5M dialogues, average 7.6 turns, average utterance length 16.1, lexical diversity (MTLD) 68.0. Comparison rows: DailyDialog 13K / 7.9 / 14.6 / 63.0; ProsocialDialog 58K / 5.7 / 20.0 / 60.2.
- **Emotion coverage:** 385K conversations generated from 1.7K unique emotion descriptions in xReact tails (§3.2).
- **Cost:** about $0.02 per triple-to-dialogue with `text-davinci-002`; 10 queries take under 2 minutes including the filtration pipeline (§3.2 "Cost & Time-Efficient"). The paper does not report a total spend.
- **COSMO results:** on DailyDialog, unseen at training, COSMO-3B wins overall 72% vs BlenderBot-3B 28%, 86% vs GODEL 14%, 71% vs Koala-7B 29%, 58% vs Vicuna-7B 42%, and 55% vs the human ground-truth responses 45% (Table 5). On BlendedSkillTalk, BlenderBot's own training domain, COSMO-3B wins overall 64% vs 36% (Table 6). On SODA itself, COSMO-11B is 53% vs GPT-3.5 and 50% vs ChatGPT overall (Table 7).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| COSMO | 3B, 11B | distill-SFT | Backbone | LM-adapted T5, trained with the T5X library | arXiv:2212.10465v3 §4 | verified (2026-09-18) | no ablation reported |
| COSMO | 3B, 11B | distill-SFT | Input format | narrative n, speaker instruction i, dialogue context c concatenated with `<SEP>`; utterances joined with `<TURN>`; target is next response p(r\|n,i,c) | arXiv:2212.10465v3 §4 | verified (2026-09-18) | no ablation reported |
| COSMO | 3B, 11B | distill-SFT | Context dropout | narrative dropped 30% of the time, role instruction 50% of the time | arXiv:2212.10465v3 §4 | verified (2026-09-18) | stated as serving robustness on datasets without contexts; no ablation reported |
| COSMO | 3B, 11B | distill-SFT | Training data | SODA plus ProsocialDialog reformatted to the SODA schema | arXiv:2212.10465v3 §4, App. C | verified (2026-09-18) | cited to Kim et al. 2022a for handling sensitive contexts without harming other datasets |
| SODA generation | — | data synthesis | Teacher, generation cost | `text-davinci-002`; ~$0.02 per dialogue | arXiv:2212.10465v3 §2, §3.2 | verified (2026-09-18) | no ablation reported |

Optimizer, learning rate, batch size, epochs, and sequence length for COSMO are not reported in the paper or its appendices.

## Findings relevant to generality
- **Out-of-domain transfer is the paper's main generality claim.** COSMO-3B was never trained on DailyDialog or BlendedSkillTalk and wins head-to-head on both, including against the model trained on BlendedSkillTalk (§5.1, §5.2, Tables 5–6).
- **Scale of the student versus the data source:** COSMO-3B outperforms baselines despite 1.5M dialogues versus BlenderBot's 1.5B Reddit comments and GODEL's 551M Reddit threads, and 3B versus 7B parameters for Koala and Vicuna (§5.1).
- **Narrowing acknowledged by the authors:** the pipeline is restricted to social x-relations, leaving physical and event-centered commonsense to future work (§2.2, footnote 3); multi-party conversations are generated by the pipeline but discarded in favor of dyadic dialogue (§3.1, footnote 5); the teacher choice is stated to affect the types of dialogue produced (§7 Limitations).
- **Long context: not addressed.** SODA conversations are single-session, 4 to 20 turns, average 7.6 turns (§3.1, Table 2). The paper reports no multi-session, cross-session-memory, or long-horizon dialogue construction. [[conversation-chronicles]] is the follow-on work that builds multi-session episodes; this card does not carry evidence for that setting.
- **Negative feedback:** not a training signal here. Filtering removes bad generations rather than training against them; the only negative-facing component is the ProsocialDialog corpus added to COSMO's training mix, which supplies constructive feedback as ordinary supervised targets (§4).

## Connections
- [[self-instruct]] — earlier seed-and-expand synthesis from the same research community; SODA replaces seed instructions with knowledge-graph triples.
- [[prosocial-dialog]] — added to COSMO's training mixture for sensitive contexts (§4, App. C).
- [[conversation-chronicles]] — extends single-session social dialogue toward multi-session episodes; SODA is the single-session reference point.
- [[ultrachat-pipeline]], [[baize-construction]] — topic-driven and self-chat dialogue synthesis; SODA's contrast condition is uncontextualized teacher sampling, not these pipelines.
- [[camel]] — role-play-based two-agent dialogue synthesis.
- [[openassistant]] — human-authored multi-turn comparison point.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2212.10465 (arXiv v3, 2023-10-23)
- Corrections to the previous card version:
  - "Teacher model: GPT-3.5-turbo (Dec 2022)" → the teacher is `text-davinci-002` GPT-3.5 (§2, footnote 2).
  - "avg 20 tokens per turn" → average utterance length is 16.1 (Table 2); 20.0 is ProsocialDialog's value in the same table.
  - "Atomic 10X ... 10M commonsense triples" → the paper states SODA is built on 1.5 million Atomic10x triples and gives no size for Atomic10x itself (§3.2).
  - "Sample 1.5M triples balanced across relations (xWant, xNeed, xEffect, xReact)" → relation shares are unequal: xIntent 23%, xAttr 18%, xEffect 17%, xNeed 7% (Table 3).
  - "dialog must be 3+ turns" → the filter removes conversations with fewer than four or more than twenty turns (§3.1).
  - "No toxic content (GPT-3.5 internal safety + follow-up toxicity classifier)" → two named filters, Canary and the Rewire API, plus a separate GPT-3.5 zero-shot commonsense filter (§3.1).
  - "1.5M dialogs" as the exact count → 1,486,896 after filtering, from 2.2M generated (§3.1).
  - "COSMO-3B outperforms BlenderBot-3B and GODEL on conversational quality" kept, with the head-to-head numbers now attached (Table 5).
- Removed as unsupported by the source:
  - "Cost: ~$10K API" — the paper reports only ~$0.02 per dialogue, no total.
  - "Turn-count distribution: median 7–8 turns, max ~12" — the paper reports only the mean 7.6 and the 4–20 filter bounds.
  - "Diversity (unique bigrams): higher than Topical-Chat, similar to real dialog corpora" — diversity is measured with MTLD, and Topical-Chat is not a comparison dataset.
  - "SODA dialogues rated higher on 'emotionally grounded' than UltraChat / Baize" — no such comparison or metric exists in the paper.
  - "Strong transfer to out-of-distribution social dialog benchmarks (DailyDialog, EmpatheticDialogues)" — DailyDialog and BlendedSkillTalk are the out-of-domain evaluations; EmpatheticDialogues appears only as a statistics row in Table 2.
  - "feel substantially more human and emotionally grounded than topic-only prompted dialogs (UltraChat-style)" — the contrast condition in §3.3 is uncontextualized GPT-3.5 sampling, not UltraChat.
  - "Narrative-grounded dialog may become formulaic — 'PersonX said X because...' patterns leak into students" — not a risk the paper states or measures.
  - "Atomic-bias: non-Western social contexts underrepresented" — the paper discusses name-based demographic bias and US SSN name lists, not a stated Western-worldview finding about Atomic10x.
  - "$10K", "Yejin Choi lineage is the through-line" framing, and "first open commonsense-conversational model" — lineage and firsts beyond "first publicly available, million-scale high-quality social dialogue dataset" are not claimed by the paper.
- Not reported by the source: COSMO optimizer, learning rate, batch size, epochs, sequence length; total API spend; per-relation dialogue counts beyond Table 3 percentages.
