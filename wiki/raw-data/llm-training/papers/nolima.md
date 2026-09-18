<!-- scope: NoLiMa needle-in-a-haystack benchmark with minimal question-needle lexical overlap: needle set, haystack filtering, effective-length metric, results for 13 LLMs, CoT / reasoning-model and literal-match ablations
     see-also: [[ruler]], [[babilong]], [[lost-in-the-middle]], [[helmet]], [[needle-in-haystack-data]], [[michelangelo]], [[context-length-alone-hurts]]
-->

# NoLiMa: Long-Context Evaluation Beyond Literal Matching
- **Core Insight:** When the question shares almost no words with the needle (ROUGE-1 precision 0.069 vs 0.905 for vanilla NIAH, Table 1), 11 of 13 LLMs that claim at least 128K context score at or below half of their short-context base score at 32K, and GPT-4o falls from 99.3 to 69.7 (Abstract; §4.4; Table 3).
- **Guideline:** When a needle test is used to claim long-context ability, remove literal overlap between question and needle, because with overlap Llama 3.3 70B stays at 98.5 at 32K while the same facts asked through a one-hop association score 56.2 (Table 6).
- **Authors:** Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan Rossi, Seunghyun Yoon, et al. (LMU Munich; Munich Center for Machine Learning; Adobe Research)
- **Year:** 2025 (arXiv v1 2025-02; ICML 2025, PMLR 267)
- **URL:** https://arxiv.org/abs/2502.05167 (data and code: https://github.com/adobe-research/NoLiMa)
- **Source type:** paper
- **Relevant topics:** long-context evaluation, effective context length, literal matching, latent associative reasoning, distractors, chain-of-thought, reasoning models

## Abstract
Needle-in-a-haystack (NIAH) tests ask a model to retrieve a relevant fact (the needle) from long irrelevant text (the haystack). In existing variants the question and the needle often share words, so models can locate the needle by lexical matching. NoLiMa builds a needle set in which question and needle have minimal lexical overlap, so the model must infer a latent association (for example, Semper Opera House is in Dresden) to find the needle. The authors evaluate 13 LLMs that claim at least 128K tokens. Scores are high below 1K tokens and fall as length grows; at 32K, 11 models drop below 50% of their short-length baselines, and GPT-4o goes from 99.3% to 69.7%. The authors attribute the decline to attention having more difficulty retrieving information in longer contexts when literal matches are absent. Chain-of-thought prompting and reasoning models do not remove the decline.

## Key Contributions
- A quantification of question-context literal overlap across benchmarks with ROUGE precision (Table 1).
- A needle set of 28 keyword pairs expanded to 58 question-needle pairs, with one-hop and two-hop associations and default vs inverted word order (§3; §4.1; App. A).
- A two-stage haystack filtering pipeline that removes lexical distractors and unintended answer candidates (§3.1; Fig. 1).
- A base score and effective-length definition that separates task difficulty from length generalization (§4.3).
- Analyses of hops, word order, needle depth, CoT prompting, reasoning models (NoLiMa-Hard), literal-match ablations, and distractors (§4.4).

## Key Figures/Tables to Study
- **Table 1** — ROUGE-1/2/L precision between question and context for ∞Bench, RULER, HELMET, NIAH, BABILong, NoLiMa.
- **Table 3** — claimed vs effective length and scores at 1K-32K for 13 models.
- **Fig. 2** — one-hop vs two-hop and default vs inverted order (normalized scores).
- **Figs. 3-4** — full-depth sweep vs aligned last-2K sweep.
- **Tables 4-6** — CoT, NoLiMa-Hard with reasoning models, and literal-match ablations.
- **Table 10 (App. E)** — 64K and 128K results and newer models.

## Technical Details
### Needle set (§3; §4.1; App. A)
- Template example: needle "Actually, [CHAR] lives next to the Wn." with Wn a building or landmark (Semper Opera House); question "Which character has been to Wq?" with Wq a city (Dresden); the two-hop form asks about the state (Saxony) (§3; Table 2). Some needles use commonsense links, for example "vegan" vs "cannot eat fish-based meals" (§3).
- 5 needle groups, each with a default order ("... [CHAR] ... Wn") and an inverted order ("Wn ... [CHAR] ..."); 2-6 keyword sets per group; 28 keyword pairs; 58 question-needle pairs (§4.1; App. A).
- Constraints: associations must be solvable without distractor text; character names are randomized from a diverse pool and excluded if they occur in the haystack; Wn must be uniquely associated with Wq (the authors reject "Cambridge" because it is not unique to the UK); needles start with a preface phrase such as "Actually," or "In 2013," (§3; App. A).
- Literal overlap (ROUGE-1 precision, the share of question tokens found in the relevant context): ∞Bench QA 0.966, ∞Bench MC 0.946, vanilla NIAH 0.905, RULER QA 0.809, HELMET RAG 0.689, RULER S-NIAH 0.571, BABILong 0K 0.553, NoLiMa 0.069 (Table 1; fn. 2).

### Haystack construction (§3.1; §4.1)
- Distractor filtering: Contriever embeddings of all haystack words; the top-20 most similar words per Wq are inspected manually, and sentences containing flagged words are removed (§3.1).
- Answer-candidate filtering: Llama 3.3 70B Instruct reads 1,000-character chunks with an 800-character stride (about 250 tokens) with a 4-shot prompt (2 N/A examples) and flags possible answers; one author reviews flagged cases; repeated until no removals (§3.1; fn. 3-4). In a control test with each needle placed in 100 random chunks, the filter model scores 99.8% (§4.2).
- Haystacks: 10 open-license books of at least 50K tokens each; random snippets under 250 tokens are concatenated until more than 2K lines (more than 60K tokens), which limits memorization of the books (§4.1).
- Tests: each needle is placed 26 times at equal intervals; 5 haystacks × 58 pairs × 26 placements = 7,540 tests per context length (§4.1).

### Metric and inference settings (§4.3; App. B-C)
- Lengths: 250, 500, 1K, 2K, 4K, 8K, 16K, 32K. Accuracy: the output contains the correct character name (§4.3).
- Base score: for each question-needle pair, average over the 5 haystacks, take the maximum over the 250, 500, and 1K lengths, then average over pairs (§4.3).
- Effective length: the largest tested length whose score exceeds 0.85 × base score (RULER instead uses a fixed 85.6 threshold from Llama 2 at 4K) (§4.3). Worked example from Table 3: GPT-4o base 99.3 gives threshold 84.4; 8K scores 89.2 (above) and 16K scores 81.6 (below), so the effective length is 8K.
- Decoding: greedy for instruction-tuned models; default sampling for GPT-o1 and GPT-o3 Mini; top-p 0.95 and temperature 0.6 for DeepSeek-R1-Distill-Llama-70B; reasoning models capped at 1,536 generated tokens; CoT limited to three sentences or 192 tokens (App. C). Open-weight models served with vLLM (App. B).

### Main results (Table 3; claimed / effective length, base score, 32K score)
| Model | Claimed | Effective | Base | 32K |
|---|---|---|---|---|
| GPT-4o | 128K | 8K | 99.3 | 69.7 |
| Llama 3.3 70B | 128K | 2K | 97.3 | 42.7 |
| Llama 3.1 405B | 128K | 2K | 94.7 | 38.0 |
| Llama 3.1 70B | 128K | 2K | 94.5 | 43.2 |
| Gemini 1.5 Pro | 2M | 2K | 92.6 | 48.2 |
| Jamba 1.5 Mini | 256K | <1K | 92.4 | 43.6 |
| Command R+ | 128K | <1K | 90.9 | 7.4 |
| Gemini 2.0 Flash | 1M | 4K | 89.4 | 41.0 |
| Mistral Large 2 | 128K | 2K | 87.9 | 18.8 |
| Claude 3.5 Sonnet | 200K | 4K | 87.5 | 29.8 |
| Gemini 1.5 Flash | 1M | <1K | 84.7 | 28.6 |
| GPT-4o mini | 128K | <1K | 84.8 | 13.7 |
| Llama 3.1 8B | 128K | 1K | 76.7 | 14.2 |

- The two models above half of their base score at 32K are GPT-4o (69.7/99.3) and Gemini 1.5 Pro (48.2/92.6) (derived from Table 3). The authors cite Llama 3.1 70B effective lengths of 16K on BABILong QA1 and 32K on RULER, against 2K here (§4.4).
- Two-hop questions score lower than one-hop at the same length, and the gap widens with length; inverted order is harder than default order (§4.4.1; Fig. 2). Two-hop effective lengths in App. F: GPT-4o 8K, Llama 3.3 70B 1K (Table 12).
- Beyond 32K with 11 placements instead of 26: GPT-4o 62.4 at 64K and 56.0 at a 127,500-token haystack; Gemini 2.0 Flash 33.0 and 16.4; GPT-4.1 has base 97.0, effective length 16K, and 64.7 at 128K (App. E; Table 10).

### Ablations (§4.4.2-4.4.4; Llama 3.3 70B unless stated)
- Depth (51 placements, moving average of 12): one-hop shows a lost-in-the-middle dip at 32K; for two-hop, longer contexts lower the curve at all depths, including the edges (§4.4.2; Fig. 3). In the aligned last-2K sweep the question-needle distance is the same across context lengths, yet two-hop scores fall with length; the authors conclude the drop is not explained by RoPE position encoding but by attention over more tokens (Interpretation) (§4.4.2; Fig. 3d).
- CoT (Table 4; 4K / 32K): one-hop 90.3 / 56.2 without CoT and 95.6 / 60.6 with CoT; two-hop 70.7 / 25.9 without and 82.4 / 34.3 with. Two-hop with CoT stays below one-hop without CoT at every length (derived from Table 4; the authors write that two-hop with CoT "barely" reaches one-hop without CoT).
- NoLiMa-Hard (10 hardest of 58 pairs; base / 32K): GPT-o1 99.9 / 31.1; GPT-o3 Mini 98.8 / 18.9; DeepSeek-R1-Distill-Llama-70B 99.9 / 20.7; Llama 3.3 70B 98.3 / 8.9 without CoT and 97.1 / 10.1 with CoT (Table 5).
- Literal match (Table 6; 8K / 16K / 32K): Direct questions naming Wn 98.3 / 98.5 / 98.5; one-hop 84.1 / 73.2 / 56.2 vs 98.7 / 97.4 / 93.1 as multiple choice with the four candidate names; two-hop 57.4 / 42.7 / 25.9 vs 96.3 / 94.6 / 87.2.
- Distractors: one sentence containing Wq ("There was an article about Wq in the daily newspaper."), placed at least 20% of the context length from the needle and within the 20%-80% span (App. D). Base scores fall to 93.8 (GPT-4o) and 84.4 (Llama 3.3 70B), and GPT-4o's effective length becomes 1K (§4.4.4; Fig. 5).

## Findings relevant to generality and long context
- **NIAH scores overstate long-context use.** The same model is near 98% at 32K when the question repeats the needle keyword and 56.2% (one-hop) or 25.9% (two-hop) when it does not (Table 6). Result (single study, Llama 3.3 70B).
- **Claimed vs effective length.** No model in Table 3 has an effective length above 8K, including models claiming 1M-2M tokens (Table 3); GPT-4.1 reaches 16K (Table 10). Result (single study; effective length depends on the 0.85 × base threshold).
- **Inference-time reasoning is a partial fix.** CoT raised Llama 3.3 70B two-hop accuracy at 32K from 25.9 to 34.3, and reasoning models stay below 50% of base at 32K on NoLiMa-Hard (Tables 4-5). Result (single study).
- **Implication for retrieval.** The authors expect the effect to extend to search and RAG, where a relevant document can have a lexical gap with the query while retrieved distractors overlap it (§5). Interpretation; not tested.
- Training interventions are not studied.

## Connections
- [[ruler]] — source of the effective-length idea; NoLiMa replaces RULER's fixed threshold with 0.85 × base score.
- [[babilong]] — another reasoning-in-a-haystack benchmark; its 0K question overlap is 0.553 (Table 1).
- [[lost-in-the-middle]] — the depth dip NoLiMa observes for one-hop questions at 32K.
- [[helmet]] — application benchmark whose RAG tasks keep 0.689 question overlap (Table 1).
- [[needle-in-haystack-data]] — vanilla NIAH construction that NoLiMa modifies.
- [[michelangelo]] — a NIAH extension toward latent structure queries, cited in NoLiMa §2; [[context-length-alone-hurts]] — a later study of length effects when retrieval succeeds.
- [[induction-heads]] — cited by the authors for attention's strength at recalling repeated patterns (§1-§2).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2502.05167 (arXiv v3, 2025-07-09; submission history v1 2025-02-07, v2 2025-03-26).
- Audit claims not found in the source: none. The audit's "effective length ... scoring >=85%" is stated in the source as a score that exceeds the threshold (§4.3).
- Inconsistencies inside the source: §4.4 prose gives Llama 3.1 70B "42.7% vs. 94.3% base score" at 32K, while Table 3 lists 43.2 and 94.5 for Llama 3.1 70B (42.7 and 97.3 belong to Llama 3.3 70B). §4.2 prose lists the model set with counts ("five closed-source ... seven open-weight") that do not match the 13 rows of Table 3; this card follows the tables.
- Not reported by the source: training experiments; results for earlier arXiv versions (v1 was not checked).
