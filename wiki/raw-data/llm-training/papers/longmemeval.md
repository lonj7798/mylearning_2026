<!-- scope: LongMemEval (arXiv Oct 2024; ICLR 2025) — 500 human-curated questions over length-configurable user-assistant chat histories (LongMemEval_S ~115K tokens, LongMemEval_M 500 sessions), a pilot on commercial memory assistants and long-context LLMs, and an indexing/retrieval/reading study of memory designs
     deps: [[locomo]]
     see-also: [[chroma-context-rot]], [[beam-10m-conversations]], [[lost-in-the-middle]], [[ultrachat-construction]], [[baize-construction]]
-->

# LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
- **Core Insight:** Answering from only the evidence sessions, GPT-4o scores 0.870 accuracy; reading the full LongMemEval_S history (~115K tokens) it scores 0.606 (30.3% drop), and Llama 3.1 70B Instruct drops from 0.744 to 0.334 (55.1%) (Fig. 3b).
- **Guideline:** When building a retrieval memory for a chat assistant, store rounds rather than whole sessions and expand each key with LLM-extracted user facts, because on LongMemEval_M this key expansion raised recall@k by 9.4% and final accuracy by 5.4% on average across three reader models (§5.2-5.3; Table 3).
- **Authors:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu (UCLA, Tencent AI Lab Seattle, UC San Diego)
- **Year:** 2024 (arXiv v1 2024-10; v2 2025-03; ICLR 2025)
- **URL:** https://arxiv.org/abs/2410.10813
- **Source type:** paper
- **Relevant topics:** long-term conversational memory, long-context evaluation, retrieval-augmented generation, temporal reasoning, knowledge updates, abstention, self-chat data synthesis, LLM-as-judge

## Abstract
LLM chat assistants have added memory components, but their long-term memory in sustained interactions is not well evaluated. LongMemEval tests five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. It has 500 curated questions embedded in freely scalable user-assistant chat histories. Commercial chat assistants and long-context LLMs show a 30% accuracy drop on it. The authors then describe memory design as three stages (indexing, retrieval, reading) and propose session decomposition, fact-augmented key expansion, and time-aware query expansion, which improve memory recall and question answering.

## Key Contributions
- Seven question types covering five abilities, with 30 false-premise abstention questions (§3.2).
- A history compilation method in which evidence sessions are inserted into sampled filler sessions, so history length is configurable (§3.2; App. A.2).
- A pilot showing drops for ChatGPT and Coze memory and for five long-context LLMs (§3.4; Fig. 3).
- A unified key-value view of memory systems with four control points: value, key, query, reading strategy (§4; Table 2).
- Ablations on each control point on LongMemEval_M (§5; Tables 3-4; Figs. 5-6).

## Key Figures/Tables to Study
- **Figure 2** — question construction, evidence-session self-chat, history construction.
- **Figure 3** — commercial systems vs offline reading; long-context LLMs on Oracle vs S.
- **Table 3** — recall/NDCG and end-to-end QA for key designs with round or session values.
- **Table 4** — time-aware query expansion on the temporal-reasoning subset.
- **Figure 14** — error split into retrieval vs generation failures.
- **Table 8** — five additional LLMs, including collapse of 1B/3B models on long histories.

## Technical Details
### Data construction
- Attribute ontology: 164 user attributes in five categories (demographics, lifestyle, situational context, life events, belongings) (§3.2; App. A.1 Table 5).
- For each attribute, Llama 3 70B Instruct writes a user background paragraph; it proposes seed question-answer pairs, and GPT-4o also proposes temporal-reasoning, multi-session and single-session-preference questions (§3.2 fn. 1; App. A.1).
- About 1,000 questions were generated per question type; experts filtered and rewrote them, with a final yield of about 5%. Answers were manually decomposed into evidence statements with optional timestamps (App. A.1).
- Evidence sessions: one self-chat per evidence statement with Llama 3 70B Instruct as user and assistant; the user is instructed to convey the evidence indirectly in concise 1-2 sentence messages; at most 10 rounds (App. A.1, Fig. 8).
- Experts inspected every session for evidence presence, no leakage of other evidence, colloquial time mentions, and a natural ending; roughly 70% of sessions were edited (App. A.1).
- Filler sessions: 25% ShareGPT, 25% UltraChat, 50% simulated sessions from other non-conflicting attributes; sessions are shuffled with the evidence sessions and assigned timestamps in order; without predefined timestamps, dates fall in May 2023 (App. A.2).
- Settings: LongMemEval_S about 115K tokens per question (about 50 sessions), LongMemEval_M 500 sessions (about 1.5M tokens) (§3.2; §3.4). Most questions need evidence from multiple sessions, up to six (§3.2; Fig. 9).
- Annotation: three expert annotators; about 400 hours for dataset construction and 150 hours for the commercial-system study (Ethics statement).
- Release: `longmemeval_s`, `longmemeval_m`, `longmemeval_oracle` files with 500 instances each; `_abs` question ids mark abstention; evidence turns carry `has_answer: true`; `sample_haystack_and_timestamp.py` rebuilds histories; a cleaned version of the history sessions was released in 2025-09 (GitHub README).

### Evaluation
- QA is judged by prompt-engineered gpt-4o-2024-08-06 with type-specific prompts (§3.3; App. A.4 Fig. 10). On 30 questions per type, judge accuracy against experts averages 0.98 (GPT-4o answers) and 0.97 (Llama-3.1-8B answers), with a minimum of 0.90 (App. A.4 Table 6).
- Retrieval: Recall@k and NDCG@k from the annotated evidence locations (§3.3).

### Pilot results (§3.4; Fig. 3; App. B)
- Commercial systems: 97 questions, 3-6 session histories, annotators chatting through the web interface in the first two weeks of August 2024. Offline reading with GPT-4o 0.9184; ChatGPT (GPT-4o) 0.5773, ChatGPT (GPT-4o-mini) 0.7113; Coze (GPT-4o) 0.3299, Coze (GPT-3.5-turbo) 0.2474 (Fig. 3a). Abstention was not tested (App. B).
- Failure modes: ChatGPT modified recorded facts when compressing history; Coze often failed to record indirectly given information (§3.4; App. B).
- Long-context LLMs, Oracle → S, no Chain-of-Note: GPT-4o 0.870 → 0.606; Llama 3.1 8B 0.710 → 0.454; Phi-3 128k 14B 0.702 → 0.380; Phi-3.5 Mini 4B 0.660 → 0.342. With Chain-of-Note, GPT-4o 0.924 → 0.640 (Fig. 3b).
- Llama-3.2-3B-Instruct: 0.522 (Oracle, direct) vs 0.008 (S, direct); round-level RAG with fact keys gives 0.508 on S (App. E.1 Table 8).

### Memory design results (§5; LongMemEval_M)
- Setup: readers GPT-4o, Llama 3.1 70B and 8B Instruct; retriever Stella V5 1.5B; Llama 3.1 8B extracts summaries, keyphrases, facts and timestamped events; retrieved items are sorted by time; CoN and JSON format by default; greedy decoding, 800 max tokens (§5.1; App. D).
- Value: rounds beat sessions with GPT-4o as reader; summaries or facts as values lower accuracy except on multi-session questions. Llama 3.1 8B accuracy drops beyond about 3K retrieved tokens, GPT-4o keeps improving past 20K (§5.2; Fig. 5).
- Key (value = round): K = V recall@5 0.582, GPT-4o top-10 QA 0.670; K = V + fact recall@5 0.644, QA 0.720. Summaries, keyphrases or facts used alone as keys "do not enhance the memory recall performance" (§5.3; Table 3). Merging at the ranking stage instead of the key underperforms (App. E.3 Table 10).
- Query: time-aware expansion raises temporal-reasoning recall by 11.3% (round values) and 6.8% (session values) when GPT-4o infers the time range; Llama 3.1 8B often assigns time ranges to questions that have none (§5.4; Table 4; App. E.4 Table 11).
- Reading: under oracle retrieval, a suboptimal reading strategy costs up to 10 absolute points for GPT-4o; JSON helps consistently only with CoN (§5.5; Fig. 6).
- Error analysis (best design, top-10): correct retrieval with a wrong answer is 15-19% of all instances and 40-50% of errors; correct retrieval is needed for a correct answer about 90% of the time (App. E.5; Fig. 14).

## Findings relevant to generality, negative feedback, long context, and synthetic data
- **Evidence-only vs full-history gap.** Long-context LLMs lose 30%-60% from Oracle to S, with or without CoN (§1; §3.4). Result (single study, five models). The authors expect larger drops as histories grow beyond ~50 sessions (§3.4). Interpretation.
- **Small models and long histories.** Llama-3.2 1B/3B score at or near 0.01 when reading S directly but 0.31-0.51 with round-level retrieval (Table 8). Result (single study).
- **Negative and conflicting items.** 30 false-premise questions require "I don't know" (§3.2). Knowledge-update questions contain an earlier fact superseded by a later one; the judge accepts an answer that mentions old information if the updated answer is given (App. A.4 Fig. 10). Simulated filler sessions are built from other, non-conflicting attributes, and the filler pool is chosen to have similar topic or format to the evidence sessions (App. A.2). These are evaluation items; the paper trains no model.
- **Synthetic-data construction and quality control.** Teacher/generator: Llama 3 70B Instruct (backgrounds, self-chat) and GPT-4o (some question proposals); indirect evidence placement is an explicit prompt constraint; quality control is expert filtering (~5% yield) and ~70% session edits (App. A.1).

## Connections
- [[locomo]] — earlier human-human memory benchmark that, per this paper, does not test assistant-side recall or knowledge updates (§1; Table 1).
- [[beam-10m-conversations]] — later benchmark reaching 10M tokens and adding contradiction resolution, event ordering and instruction following.
- [[chroma-context-rot]] — uses LongMemEval_s prompts for a focused (~300 tokens) vs full (~113K tokens) comparison.
- [[ultrachat-construction]] — UltraChat, one of the filler-session sources; [[baize-construction]] — self-chat method cited for evidence sessions.
- [[lost-in-the-middle]] — cited for weakened use of long inputs.
- [[msc-beyond-goldfish-memory]] — MSC, compared in Table 1.
- [[memoryagentbench]] and [[memagent]] — later memory-agent benchmark and method for comparison.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2410.10813 (arXiv v2, 2025-03-04; v1 2024-10-14) and https://github.com/xiaowu0162/LongMemEval README (file names, `_abs`, `has_answer`, history script, 2025-09 cleaned release).
- Audit claims not found in the source: (1) "half the filler sessions are simulated on the same topics as the evidence" → simulated from "other non-conflicting attributes" so that non-evidence sessions have "similar topic or format" (App. A.2). (2) "LongMemEval_S ~40-50 sessions" → the paper says ~50 sessions (§3.4); the README says ~40 sessions for Llama 3. (3) "no short-context effect reported" and "Chroma's focused vs full result" are not statements of this paper. Other audit numbers (164 attributes, 5% yield, 70% edits, 25/25/50 mix, 91.84/57.73/32.99, +9.4/+5.4, +6.8-11.3, up to 10 points, ~400 hours) match the loci above.
- Not reported by the source: exact per-type question counts in the text (Fig. 9a gives percentages only); token counts per LongMemEval_M history beyond "~1.5M".
