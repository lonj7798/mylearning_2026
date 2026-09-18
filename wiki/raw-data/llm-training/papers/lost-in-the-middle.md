<!-- scope: position-sensitivity study of long-context use (multi-document QA and JSON key-value retrieval), with architecture, query-placement, instruction-tuning, and open-domain QA analyses
     see-also: [[helmet]], [[nolima]], [[ruler]], [[babilong]], [[needle-in-haystack-data]], [[in2-film]], [[pam-qa-never-lost-in-middle]]
-->

# Lost in the Middle: How Language Models Use Long Contexts
- **Core Insight:** Accuracy depends on where the relevant document sits in the prompt: with 20 retrieved documents, GPT-3.5-Turbo scores 75.8% when the answer document is first, 53.8% when it is 10th, and 63.2% when it is 20th, while its closed-book score (no documents) is 56.1% (App. G Table 6; Table 1).
- **Guideline:** When a model is claimed to use long inputs robustly, report accuracy as a function of the relevant information's position and the best-minus-worst gap, because the authors show that an average over positions hides a U-shaped curve (§1; §2.3).
- **Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, et al. (Stanford University, UC Berkeley, Samaya AI)
- **Year:** 2023 (arXiv v1 2023-07; Transactions of the ACL)
- **URL:** https://arxiv.org/abs/2307.03172
- **Source type:** paper
- **Relevant topics:** long-context evaluation, positional bias, retrieval-augmented generation, context-window extension, instruction tuning effects, evaluation protocol

## Abstract
The paper measures how language models use information in long inputs on two tasks: multi-document question answering and synthetic key-value retrieval. The authors change the position of the relevant information while holding the rest of the input fixed. Performance is often highest when the relevant information is at the beginning or the end of the input and lowest when it is in the middle; this holds for models built for long contexts. The authors analyze model architecture, query placement, and instruction fine-tuning, run an open-domain QA case study, and propose position-controlled evaluation protocols for long-context models.

## Key Contributions
- A multi-document QA setup in which exactly one of k documents contains the answer, with controlled document count (10, 20, 30) and controlled answer position (§2.1; §2.3).
- A synthetic JSON key-value retrieval task with UUID keys and values that removes natural-language cues (§3.1).
- Evidence that extended-context variants perform nearly the same as their base versions when the input fits both windows (§2.3).
- Ablations on encoder-decoder vs decoder-only models, query-aware contextualization, and instruction fine-tuning (§4).
- An open-domain QA case study showing reader accuracy saturates before retriever recall (§5).

## Key Figures/Tables to Study
- **Fig. 1 / Fig. 5** — the U-shaped accuracy curve over answer position at 10, 20, and 30 documents.
- **Table 1** — closed-book and oracle accuracy per model, the two reference lines for Fig. 5.
- **Fig. 7** — key-value retrieval at 75, 140, and 300 pairs.
- **Fig. 8** — Flan-UL2 and Flan-T5-XXL inside vs beyond their training lengths.
- **Fig. 10 and App. E Fig. 16** — base vs instruction-tuned MPT-30B and Llama-2 7B/13B/70B.
- **Fig. 11** — retriever recall vs reader accuracy as the number of retrieved documents grows.
- **App. G Tables 5-7** — the exact per-position numbers behind Fig. 5.

## Technical Details
### Multi-document QA setup
- Data: 2,655 NaturalQuestions-Open queries whose annotated long answer is a paragraph; documents are Wikipedia passages of at most 100 tokens (§2.1).
- The answer document is the NaturalQuestions-annotated paragraph. The k−1 distractors are the passages most relevant to the query, retrieved by Contriever fine-tuned on MS-MARCO, that do not contain an annotated answer; they appear in decreasing relevance order (§2.1).
- Position is changed by reordering documents; length is changed by adding or removing distractors (§2.1, Figs. 3-4). The 10, 20, and 30-document settings are about 2K, 4K, and 6K tokens (Fig. 5 titles).
- Metric: accuracy, defined as whether any annotated answer string appears in the output (§2.1). Decoding is greedy (§2.2).
- Controls: an unambiguous-question subset built from AmbigQA annotations gives similar results (App. A); random Wikipedia distractors raise absolute accuracy but the position effect remains (App. B); randomly ordered distractors with an instruction saying so keep the U-shape (App. C).

### Models
- Open: MPT-30B-Instruct (8,192-token window; pre-trained on 1T tokens at 2,048 tokens, then 50B tokens at 8,192; ALiBi positions) and LongChat-13B (16K), which extends LLaMA-13B from 2,048 to 16,384 tokens with condensed rotary embeddings before fine-tuning at 16,384 (§2.2).
- Closed: GPT-3.5-Turbo (4K) and GPT-3.5-Turbo (16K), both 0613 versions; Claude-1.3 (8K) and Claude-1.3 (100K) (§2.2).
- GPT-4 (8K) is tested on 500 random 20-document examples and shows the same U-shape with higher absolute accuracy (App. D, Fig. 15). Llama-2 7B/13B/70B base and chat are tested with 20 documents after discarding 20 of 2,655 examples longer than 4,096 tokens (App. E).

### Multi-document QA results
- Closed-book / oracle accuracy: LongChat-13B (16K) 35.0 / 83.4; MPT-30B-Instruct 31.5 / 81.9; GPT-3.5-Turbo 56.1 / 88.3; GPT-3.5-Turbo (16K) 56.0 / 88.6; Claude-1.3 48.3 / 76.1; Claude-1.3 (100K) 48.2 / 76.4 (Table 1). Oracle means the model sees only the answer document.
- 20 documents, answer at positions 1 / 5 / 10 / 15 / 20: GPT-3.5-Turbo 75.8 / 57.2 / 53.8 / 55.4 / 63.2; Claude-1.3 59.9 / 55.9 / 56.8 / 57.2 / 60.1; MPT-30B-Instruct 53.7 / 51.8 / 52.2 / 52.7 / 56.3; LongChat-13B (16K) 68.6 / 57.4 / 55.3 / 52.5 / 55.0 (App. G Table 6).
- 30 documents, GPT-3.5-Turbo (16K): 73.4 at position 1, 50.5 at position 10, 63.7 at position 30 (App. G Table 7).
- GPT-3.5-Turbo's multi-document QA accuracy "can drop by more than 20%" with position, and its worst case in the 20- and 30-document settings is below its 56.1% closed-book accuracy (§2.3).
- GPT-3.5-Turbo and GPT-3.5-Turbo (16K) curves are nearly superimposed in the 10- and 20-document settings, which fit both windows (§2.3); for example 75.8 vs 75.7 at position 1 with 20 documents (Table 6).

### Key-value retrieval
- Input: a JSON object of k pairs where keys and values are random 128-bit UUIDs; output: the value for one queried key (§3.1; Fig. 6). Settings: 75, 140, and 300 pairs (about 4K, 8K, 16K tokens), 500 examples each (§3.2; Fig. 7).
- Claude-1.3 and Claude-1.3 (100K) are nearly perfect at all three lengths; GPT-3.5-Turbo, GPT-3.5-Turbo (16K), and MPT-30B-Instruct are lowest when the key is in the middle (§3.2). At 140 pairs LongChat-13B (16K) tends to generate code to retrieve the key instead of outputting the value when the key is at the start (qualitative observation) (§3.2).

### Why the position effect appears (§4)
- **Architecture:** Flan-UL2 has a 1.9-point best-minus-worst gap on inputs within its 2,048-token training window and a U-shape beyond it; Flan-T5-XXL (512-token training) shows the same trend (§4.1; Fig. 8). The authors hypothesize that the bidirectional encoder helps (Interpretation).
- **Query-aware contextualization** (query placed both before and after the data): all models become near-perfect on key-value retrieval at 75, 140, and 300 pairs; GPT-3.5-Turbo (16K) reaches perfect accuracy at 300 pairs, against a 45.6% worst case without it (§4.2). On multi-document QA it slightly raises accuracy at position 1 and slightly lowers it elsewhere (§4.2; Fig. 9).
- **Instruction fine-tuning:** MPT-30B (base) and MPT-30B-Instruct both show the U-shape; the best-minus-worst gap is nearly 10 points for the base model and around 4 points after instruction tuning (§4.3; Fig. 10).
- **Scale (Llama-2):** 7B models are only recency-biased; 13B and 70B show both primacy and recency bias. The 13B base model has a 20-point gap, reduced to 10 points by SFT and RLHF; the 70B trends change little with fine-tuning (App. E; Fig. 16).

### Open-domain QA case study (§5)
- Retriever: Contriever fine-tuned on MS-MARCO over Wikipedia; readers receive the top-k documents; metrics are retriever recall and reader accuracy (§5).
- Reader accuracy saturates before retriever recall. Going beyond 20 documents (to 50) adds about 1.5 points for GPT-3.5-Turbo and about 1 point for Claude-1.3 (§1; §5; Fig. 11). The authors suggest reranking relevant documents toward the start or truncating the ranked list (§5).

## Findings relevant to generality and long context
- **A larger window is not better use.** Extended-window variants (GPT-3.5-Turbo 4K vs 16K, Claude-1.3 8K vs 100K) had nearly identical accuracy on inputs that fit both windows (§2.3). Result (single study).
- **Measurement protocol.** The authors state that a claim of robust long-context use requires minimal difference between best- and worst-case position performance (§1). A single-position needle test does not provide this.
- **Effect of post-training.** Instruction tuning reduced, but did not remove, the position gap for MPT-30B (about 10 to about 4 points) and Llama-2 13B (20 to 10 points), and did not change it at Llama-2 70B (§4.3; App. E). Result (single study). The authors also report that base models already show the U-shape, so instruction fine-tuning is not necessarily its cause (Fig. 10 caption).
- **Pre-training hypothesis.** The authors hypothesize that base models learn to use the start of the context from similarly formatted internet text such as StackOverflow questions and answers (§4.3). Interpretation; not tested.
- **Length generalization.** Encoder-decoder robustness held only within the training-time sequence length (§4.1). Result (single study).

## Connections
- [[helmet]] — extends the position analysis to 128K tokens with six needle depths on JSON KV and RAG tasks (HELMET App. E.4).
- [[nolima]] — reproduces a lost-in-the-middle dip at 32K for one-hop questions and reports that two-hop questions degrade at all depths.
- [[ruler]] and [[needle-in-haystack-data]] — synthetic retrieval tests with controllable input length.
- [[babilong]] — reports the lowest GPT-4-Turbo QA1 accuracy when facts sit at depth 50 (BABILong App. K).
- [[longchat]] — LongChat-13B (16K), tested here, extended with condensed rotary embeddings; [[position-interpolation]] — a related rotary-interpolation extension method.
- [[alibi]] — the position scheme of MPT-30B-Instruct.
- [[in2-film]] and [[pam-qa-never-lost-in-middle]] — later training-data methods that target position sensitivity.
- [[long-context-llms-meet-rag]] and [[rag-or-long-context-self-route]] — later studies of retrieval count vs long-context reading.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2307.03172 (arXiv v3, 2023-11-20; submission history v1 2023-07-06).
- Audit claims not found in the source: none. The audit phrase "open-domain QA saturates around 20 retrieved docs" is a paraphrase; the paper states that using more than 20 (50 instead of 20) retrieved documents improves accuracy by about 1.5 points for GPT-3.5-Turbo (§1; §5).
- Not reported by the source: training experiments; results for GPT-4 on key-value retrieval (cost, fn. 6).
