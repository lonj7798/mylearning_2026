<!-- scope: BEAM (arXiv Oct 2025; ICLR 2026) — plan-driven generation of coherent single-user user-assistant conversations from ~100K to 10M tokens across 19 domains, 2,000 human-validated probing questions over ten memory abilities with nugget scoring, and the LIGHT memory scaffold (episodic retrieval, working memory, scratchpad)
     deps: [[longmemeval]], [[locomo]]
     see-also: [[msc-beyond-goldfish-memory]], [[llama-4]], [[memoryagentbench]], [[chroma-context-rot]], [[rag-or-long-context-self-route]]
-->

# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs
- **Core Insight:** On BEAM, average probing-question scores for long-context baselines fall from 0.239-0.280 at 100K tokens to 0.104-0.133 at 10M tokens (where each model reads only the most recent segment that fits its window), and LIGHT reaches 0.192-0.266 at 10M (Table 1; §4.1).
- **Guideline:** When a conversation exceeds the model's context window, combine turn-level retrieval with a filtered running scratchpad instead of reading only the recent window, because at 10M tokens removing retrieval or scratchpad filtering from LIGHT (Qwen2.5-32B-AWQ) lowers the average from 0.238 to 0.153 and 0.155 (App. C.1 Table 8).
- **Authors:** Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J. Ross Mitchell (University of Alberta, University of Massachusetts Amherst)
- **Year:** 2025 (arXiv v1 2025-10; v2 2026-02; ICLR 2026)
- **URL:** https://arxiv.org/abs/2510.27246
- **Source type:** paper
- **Relevant topics:** long-term conversational memory, synthetic long-conversation generation, long-context evaluation, retrieval-augmented generation, contradiction resolution, knowledge updates, abstention, nugget-based LLM judging

## Abstract
Existing long-term memory benchmarks often lack narrative coherence, cover narrow domains, and test simple recall. The authors present a framework that automatically generates long (up to 10M tokens), coherent, topically diverse conversations with probing questions for many memory abilities, and use it to build BEAM: 100 conversations and 2,000 validated questions. They also propose LIGHT, which gives an LLM a long-term episodic memory, a short-term working memory, and a scratchpad of salient facts. On BEAM, LLMs with 1M-token windows, with or without retrieval augmentation, degrade as dialogues lengthen. LIGHT improves results across models by an average of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM, and an ablation shows each component contributes.

## Key Contributions
- A four-stage generator: conversation plans with sub-plans, user turns from batched plan bullets, assistant turns with question-detection and follow-up modules, and plan-derived probing questions (§2.2-2.3; Fig. 1).
- Ten memory abilities, three introduced here: instruction following, event ordering, contradiction resolution (§2.2; App. B.1 Table 2).
- Nugget-based scoring for nine abilities and Kendall tau-b for event ordering (§2.4).
- LIGHT: episodic retrieval over key-value indices, recent-turn working memory, and a question-filtered scratchpad (§3; Fig. 2).
- Released code, data and evaluation scripts; datasets on Hugging Face as BEAM (128K/500K/1M) and BEAM-10M (fn. 1; GitHub README).

## Key Figures/Tables to Study
- **Figure 1** — generation pipeline from seed to validated probes and nuggets.
- **Table 1** — Vanilla (long-context), RAG and LIGHT per ability, four backbones, four lengths.
- **Table 3 / Table 6** — conversation statistics and batching configuration per size.
- **Table 7** — decile position of evidence for probing questions.
- **Table 8** — ablation of LIGHT components.
- **Tables 9-10, 13** — retrieval depth, dense vs sparse retrieval, ReadAgent comparison.

## Technical Details
### Conversation generation
- Seed: domain, title, theme, subtopics; titles and themes proposed by GPT-4.1 for human-specified domains and filtered by humans for diversity; 15-20 narratives per conversation from Llama-3.3 70B (§2.2.1; App. B.3.2).
- User profile with MBTI-based traits composed from six randomly chosen types (8,008 unique profiles), a relationship graph with realism constraints, and an explicit timeline (App. B.3.2).
- 128K, 500K, 1M conversations use one plan; 10M conversations use ten interlocking plans built by sequential expansion or hierarchical decomposition, with each plan conditioned on summaries of previous plans and on future seeds (§2.2.1).
- A second GPT-4.1 pass adds three bullet points per sub-plan to target contradiction resolution, knowledge update and instruction following; the authors report that single-pass inclusion gave lower quality and coverage (§2.2.1; App. B.3.2).
- User turns: Llama-3.3 70B generates I questions per batch of plan bullets; e.g., 10M general-domain chats use 10 sub-plans, K = 10 batches, I = 9 (§2.2.2; App. B.4 Table 6).
- Assistant turns: role-play with summaries of the last M turns and of earlier turns; assistant counter-questions and user follow-ups are each capped at δ = 2 (§2.2.3). Generation temperature 0.1; other inference temperature 0 (§4.1).
- Scale: 100 chats over 19 domains (App. B.3.1); 20 chats at 128K, 35 at 500K, 35 at 1M, 10 at 10M (GitHub README). Averages per 10M chat: 10,435 user and 10,435 assistant messages, 1,528 user follow-ups (App. B.1 Table 3).

### Probing questions and scoring
- GPT-4.1-mini selects plan bullets for each ability and writes the question, a candidate answer and source message ids; a human evaluator keeps valid questions; two questions per ability are kept per conversation, 20 per conversation (§2.3-2.4).
- Annotators split each reference answer into atomic nuggets; an LLM judge scores each nugget 0, 0.5 or 1 and scores are averaged; event ordering uses Kendall tau-b after LLM event alignment (§2.4). The judge model is not named in the main text.
- Evidence position: at 10M, 0.00% of questions draw evidence from the first decile and 2.41% from the last decile (App. B.5 Table 7).
- Conversation quality: two annotators, 5-point Likert; averages 4.53 coherence, 4.57 realism, 4.64 complexity; Cohen's κ 0.70-0.78 on 20 conversations; annotators read sampled spans (first 25 turns, middle 25, last 25, random turns) rather than full conversations (§4.2; App. B.2 Table 4).

### LIGHT and baselines
- Indexing: after each turn, Qwen2.5-32B-AWQ extracts entity key-value pairs and a summary; keys are embedded with BAAI/bge-small-en-v1.5 in FAISS; values are the original dialogue segments (§3.1; §4.1).
- Scratchpad: Qwen2.5-32B-AWQ extracts salient content per dialogue pair; above 30K tokens it is compressed to a 15K-token summary by GPT-4.1-nano; at inference it is chunked with LangChain SemanticChunker and each chunk kept only if judged relevant (§3.2).
- Baselines: Vanilla feeds the full history (GPT-4.1-nano and Gemini-2.0-flash at 1M, Qwen2.5-32B-AWQ at 128K, Llama-4-Maverick-fp8); RAG retrieves the top 5 turn pairs; at 10M, Vanilla reads the largest recent segment that fits (§4.1, fn. 3).

### Results
- Average score at 100K / 500K / 1M / 10M, LIGHT: Qwen 0.311 / 0.316 / 0.309 / 0.238; Llama-4-Maverick 0.358 / 0.359 / 0.336 / 0.266; GPT-4.1-nano 0.345 / 0.335 / 0.336 / 0.226 (Table 1).
- Relative gains over Vanilla: +49.1% (Maverick) and +44.3% (GPT-4.1-nano) at 100K; +75.9% (GPT-4.1-nano) at 1M; +155.7% (Maverick) at 10M. At 10M, Gemini-2.0-flash with LIGHT (0.192) is below RAG (0.216) (§4.2; Table 1).
- All methods score highest on abstention and lowest on contradiction resolution; for example, at 1M contradiction resolution ranges 0.007-0.050 (§4.2; Table 1).
- Ablation (Qwen, 10M): w/o retrieval 0.153, w/o scratchpad 0.202, w/o working memory 0.181, w/o noise filtering 0.155, vs 0.238 (Table 8). At 100K and 1M, removing working memory raises the average (0.327 vs 0.311; 0.328 vs 0.309) (Table 8); the §4.2 prose instead reports a -1.6% change at 100K.
- Retrieval depth: K = 15 gives the highest average at all lengths (e.g., 0.334 vs 0.311 for K = 5 at 100K); at 10M, K = 10 gives 0.171 (Table 9). SPLADE beats dense retrieval at 100K-1M and is below it at 10M (0.231 vs 0.238) (App. C.2 Table 10).
- ReadAgent averages 0.206 / 0.191 / 0.186 / 0.148 vs LIGHT 0.311 / 0.316 / 0.309 / 0.238 (App. C.5 Table 13).

## Findings relevant to generality, negative feedback, long context, and synthetic data
- **A 1M window does not remove degradation.** GPT-4.1-nano (1M window) Vanilla, given the full history, averages 0.239 at 100K and 0.191 at 1M (§4.1; Table 1). Result (single study; the number of runs is not stated).
- **Contradictions and outdated facts.** Contradiction and knowledge-update content is injected into plans on purpose (§2.2.1). Error analysis reports that models answer from the old value when both old and new values are retrieved, because retrieved documents are not in temporal order, and overweight one side of a contradiction due to position and frequency (App. G). Interpretation by the authors.
- **Abstention failures.** The main abstention failure is answering from entities, dates or concepts that are similar to but not the requested information (App. G).
- **Synthetic-data construction and quality control.** Generators: GPT-4.1 (titles, plan augmentation), Llama-3.3 70B (narratives, user turns), GPT-4.1-mini (probes); sampling temperature 0.1 for plans and turns; quality control is human probe validation and a sampled human quality rating, not full-conversation review (§2.2-2.4; §4.1; App. B.2).

## Connections
- [[longmemeval]] — earlier user-assistant memory benchmark; BEAM Table 2 lists its length as "115K, 1M", while LongMemEval reports ~1.5M tokens for LongMemEval_M.
- [[locomo]] — earlier generated two-speaker benchmark, listed at ~10K tokens in Table 2.
- [[msc-beyond-goldfish-memory]] — MSC, listed in Table 2 at ~1K tokens.
- [[llama-4]] — Llama-4-Maverick is a tested backbone; Llama-4-Scout's 10M window was not tested due to compute (fn. 3).
- [[memoryagentbench]] — later memory-agent benchmark for comparison.
- [[chroma-context-rot]] and [[rag-or-long-context-self-route]] — other studies of length-dependent degradation and retrieval vs long-context reading.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2510.27246 (arXiv v2, 2026-02-21; v1 2025-10-31) and https://github.com/mohammadtavakoli78/BEAM README (per-size chat counts, dataset names).
- Audit claims not found in the source: "well past LongMemEval_M's 1.5M" is a comparison made by the audit, not by the paper. Other audit claims (10M tokens, 100 conversations, 2,000 validated questions, 1M-window models struggle, LIGHT components, 3.5-12.69%) match the abstract.
- Not reported by the source: the LLM judge model and its agreement with humans; working-memory size z; per-question variance or multiple seeds. The benchmark labels the shortest size as 100K in Table 1 and 128K in Tables 3 and 6.
