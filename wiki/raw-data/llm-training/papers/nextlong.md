<!-- scope: NExtLong (ICML 2025) — long-context continued-pretraining data synthesized from short documents by interleaving retrieved hard-negative chunks between the chunks of one document; 128K and 512K Llama-3-8B runs, HELMET/RULER/LongBench v2 results, ablations on granularity, negative selection, position, and corpus
     deps: [[long-context-data-engineering]], [[prolong]]
     see-also: [[quest-query-centric-synthesis]], [[in-context-pretraining]], [[helmet]], [[loongrl]], [[retrieval-head]]
-->

# NExtLong: Toward Effective Long-Context Training without Long Documents
- **Core Insight:** Continued pretraining of Meta-Llama-3-8B-base at 128K on documents built by splitting a short document into meta-chunks and following each meta-chunk with its top-k most similar retrieved chunks reaches a HELMET+RULER average of 62.58, against 55.25 for Quest and 52.85 for random concatenation under the same training configuration (Table 1).
- **Guideline:** When long natural documents are scarce for the target length, synthesize training sequences whose dependent pieces are separated by semantically similar distractor chunks rather than random or self-repeated text, because top-k similar negatives gave the best result among the five selection strategies in the paper's ablation (§5.4, Fig. 7; values shown only in the figure) and short-text benchmark averages stayed within 0.1 points of the base model (Table 4).
- **Authors:** Chaochen Gao, Xing Wu, Zijia Lin, Debing Zhang, Songlin Hu (Institute of Information Engineering, CAS; University of Chinese Academy of Sciences; Tsinghua University; Xiaohongshu Inc.)
- **Year:** 2025 (arXiv v1 2025-01; ICML 2025)
- **URL:** https://arxiv.org/abs/2501.12766
- **Source type:** paper
- **Relevant topics:** long-context continued pretraining, synthetic long-context data, hard negatives as distractors, document chunking, RoPE base scaling, short-text retention, long-context evaluation

## Abstract
Long-context training is limited by the scarcity of long documents, and prior synthesis methods that concatenate short documents lack a mechanism that reinforces long-range dependency modeling. NExtLong (Negative document Extension) decomposes a document into meta-chunks and extends the context by interleaving hard-negative distractors retrieved from pretraining corpora. The model must separate the long-range dependent context from distracting content during next-token prediction. On HELMET and RULER, NExtLong improves over prior long-context synthesis methods and over leading models trained on non-synthetic long documents (Abstract). Code: github.com/caskcsg/longcontext/tree/main/NExtLong (Abstract).

## Key Contributions
- A two-stage method: Negative Document Extension (chunking plus hard-negative mining) and Long-Range Dependence Modeling with standard next-token loss (§3, Fig. 2, App. D.2 Algorithm 1).
- A controlled comparison against Standard, KNN, ICLM, and Quest synthesis at 128K with identical training settings (§4.2, Table 1).
- A 512K model trained without natural long documents, compared with ProLong-512K and Llama-3.1-8B (§4.3, Table 2) and after UltraChat SFT on LongBench v2 (§5.2, Table 3).
- Ablations on chunk granularity, negative selection, meta-chunk position, and source corpus (§5.3–5.4; App. B.1, B.3), plus an attention probe of long-range dependence (§5.1, App. B.4).

## Key Figures/Tables to Study
- **Fig. 2** and **App. D.2 Algorithm 1** — the full synthesis procedure.
- **Table 1** and **Fig. 3** — synthesis-method comparison at 128K, averaged and per length.
- **Table 2** and **Table 3** — comparison with open and closed models; LongBench v2 after SFT.
- **Table 4** — short-text benchmarks after long-context training.
- **Fig. 6 / Fig. 7 / Table 5 / Table 7** — granularity, negative-selection, position, and corpus ablations.
- **Tables 8–9** — training configurations for the 128K and 512K runs.

## Technical Details
### Synthesis procedure
- Chunking: a meta-document is split into paragraphs at newline characters; paragraphs are concatenated in order into meta-chunks until adding the next paragraph would exceed the maximum length s (§3.1.1; Algorithm 1 lines 2–20). The unit of s is not stated; App. D.1 computes k from character counts.
- Index: every document of the deduplicated pretraining corpus is split with the same granularity s, each chunk is embedded, and the embeddings are inserted into a FAISS index (§3.1.2, Eq. 2–3). The embedding model is not reported.
- Mining: for each meta-chunk m_i the top-k most similar chunks are retrieved as hard negatives n_i1..n_ik, forming the extended chunk l_i = [m_i, n_i1, ..., n_ik]; the long document is t = [l_1, ..., l_p] (Eq. 4–5). The same hard negative is not reused across meta-chunks (App. D.1). A synthesized document is kept only if its length reaches the target (Algorithm 1 line 34).
- Number of negatives: k = (T × E × w − S) / (p × s), where T is the target length, E the tokenizer encoding rate, w = 1.5 an adjustment factor, S the meta-document length in characters, p the number of meta-chunks, and s the granularity (App. D.1, Eq. 8–12).
- Loss: standard next-token prediction over all tokens of t, meta-chunks and hard negatives alike (§3.2, Eq. 6–7).
- Source corpora: Cosmopedia v2 (over 39 million generated samples) and FineWeb-Edu (1.3 trillion tokens) (§4.1). In 8 million sampled documents per corpus, 100.00% (Cosmopedia v2) and 99.19% (FineWeb-Edu) have at most 8,192 Llama-3 tokens (App. B.2, Table 6).

### Evaluation setup
- HELMET: 5 task types (synthetic recall, RAG, many-shot ICL, passage re-ranking, long-document QA), 17 sub-tasks; RULER: 13 sub-tasks; results averaged over 8K, 16K, 32K, 64K, 128K (§4.1; Table 1 caption). The "Avg." column averages the five HELMET categories and RULER.
- Baselines at 128K each synthesize 32,000 samples of 128K length (approximately 4 billion training tokens) and use the same training configuration (§4.2; App. C.1).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama-3-8B-NExtLong-128K | 8B | long-context | initial model; framework | Meta-Llama-3-8B (base); GPT-NeoX | arXiv:2501.12766v2 §4.2; Table 8 | verified 2026-09-14 | — |
| Llama-3-8B-NExtLong-128K | 8B | long-context | RoPE base | 500,000 → 200,000,000 | arXiv:2501.12766v2 §4.2; Table 8 | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-128K | 8B | long-context | tokens; batch; steps; seq length | 32,000 × 128K samples (approximately 4B tokens); 4M tokens per batch; 1,000 steps; 131,072 | arXiv:2501.12766v2 §4.2; Table 8 | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-128K | 8B | long-context | optimizer and schedule | lr 4e-5, cosine, warmup 200 iters; β1 0.9, β2 0.95; weight decay 0.1; grad clip 1.0; bfloat16 | arXiv:2501.12766v2 Table 8 | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-128K | 8B | long-context | corpus | Cosmopedia v2 + FineWeb-Edu (mixing ratio not reported) | arXiv:2501.12766v2 §4.1; App. B.3 | verified 2026-09-14 | Table 7: combined 62.58 vs FineWeb-Edu 62.04 vs Cosmopedia 59.26 |
| Llama-3-8B-NExtLong-128K | 8B | long-context | chunk granularity s | 2048 | arXiv:2501.12766v2 §5.3 | verified 2026-09-14 | Fig. 6: tested 512, 1024, 2048, 8192, 32768; 2048 best overall; 1024 best at 128K but lower at 8K/16K |
| Llama-3-8B-NExtLong-128K | 8B | long-context | negative selection; position | top-k by similarity (from 512 retrieved per meta-chunk in the ablation); meta-chunk at head | arXiv:2501.12766v2 §5.4; App. B.1 | verified 2026-09-14 | Fig. 7 (values not printed); Table 5: head 62.58, tail 60.01, random 58.95 |
| Llama-3-8B-NExtLong-128K | 8B | long-context | compute | 64 × H100, 15 h | arXiv:2501.12766v2 Table 8 | verified 2026-09-14 | — |
| Llama-3-8B-NExtLong-512K-Base | 8B | long-context | initial model; RoPE base | Llama-3-8B-ProLong-64k-Base; 128,000,000 | arXiv:2501.12766v2 Table 9 | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-512K-Base | 8B | long-context | data mixture | NExtLong-512K and NExtLong-64K samples at 1:2; 64K documents packed 8 per 512K sample with intra-document attention; 512K documents use full attention | arXiv:2501.12766v2 App. C.2 | verified 2026-09-14 | follows ProLong; no ablation reported |
| Llama-3-8B-NExtLong-512K-Base | 8B | long-context | optimizer and schedule | lr 1e-5, cosine, warmup 50 iters, 500 iters; β1 0.9, β2 0.95; weight decay 0.1; grad clip 1.0; bfloat16; seq length 524,288; batch size not reported | arXiv:2501.12766v2 Table 9 | verified 2026-09-14 | no ablation reported |
| Llama-3-8B-NExtLong-512K-Base | 8B | long-context | compute | 128 × H100, 20 h | arXiv:2501.12766v2 Table 9 | verified 2026-09-14 | — |
| Llama-3-8B-NExtLong-512K-Instruct | 8B | SFT | data; hyperparameters | UltraChat short-context SFT, following ProLong; hyperparameters not reported | arXiv:2501.12766v2 §5.2 | not reported (hyperparameters) | Table 3 LongBench v2 only |

## Findings relevant to generality, negative feedback, long context
- **Synthesis comparison at 128K.** Avg 62.58 vs Quest 55.25, Standard 52.85, KNN 50.97, ICLM 50.37; Recall 82.56 vs 69.13 (Quest); Re-rank 31.47 vs 22.35; RULER 81.50 vs 76.63 (Table 1). The gap grows with context length (Fig. 3). Result (single study, one base model).
- **512K without long documents.** Avg 65.76 vs Llama-3-8B-ProLong-512K-Base 60.34 and Llama-3.1-8B 61.07; LongQA 38.42 vs 24.68 (ProLong); Re-rank 31.27 vs 31.66 (ProLong) (Table 2). NIAH: 100% with needle positions spanning 25K–500K (App. A, Fig. 8). Both 512K models are evaluated on 128K benchmarks, following ProLong's protocol (§4.3).
- **After SFT.** LongBench v2 overall 30.4 vs 27.2 for ProLong-512K-Instruct; Long subset 26.9 vs 15.7 (Table 3).
- **Short text.** 7-task average 63.83 vs 63.75 for the base model; LAMBADA drops 75.66 → 72.06, and every concatenation baseline also lowers LAMBADA (Table 4). Result (single study).
- **Mechanism probe.** Normalized attention on the first third of the context when predicting the last token correlates positively with LongQA score (§5.1, Fig. 5); NExtLong lowers attention dependence on the last third (App. B.4, Fig. 9). Interpretation (correlation, no intervention).
- **Negatives.** "Hard negative" here means retrieved distractor text in the input, trained with ordinary next-token loss (§3.2); no loss term lowers its likelihood (negative as content). Self-repeated meta-chunks act as false negatives and lower performance (§5.4).

## Connections
- [[quest-query-centric-synthesis]], [[in-context-pretraining]] — Quest and ICLM baselines in Table 1.
- [[prolong]] — natural-long-document baseline, 512K starting checkpoint, and SFT protocol (§4.3, App. C.2, §5.2).
- [[long-context-data-engineering]] — cited as the upsampling-long-documents approach that depends on long documents (§2).
- [[helmet]] and [[ruler]] — evaluation suites; [[longbench-v2]] — post-SFT evaluation.
- [[hf-cosmopedia]] and [[fineweb]] — source corpora; [[ultrachat-construction]] — SFT data.
- [[contrastive-learning-hard-negatives]] — origin of the hard-negative idea (§1–2).
- [[llama-3]] — base model; [[loongrl]] — distractor insertion used for RL data instead of pretraining data.
- [[retrieval-head]] — attention-level account of retrieval from long context.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2501.12766 (arXiv v2, 2025-05-26; v1 2025-01-22), main text and App. A–D; HTML v2 checked for §5.4 figure values.
- Audit claims not found in the source: "tail-k (low similarity) −7 to 10%" and "random or low-similarity negatives lose 7–10%" (Fig. 7 prints no values; §5.4 text is qualitative); "meta-chunks of at most s tokens" (the unit of s is not stated); "HELMET avg" relabeled as the HELMET+RULER average (Table 1 Avg. column includes RULER).
- Not reported by the source: FAISS embedding model, corpus mixing ratio, 512K batch size, SFT hyperparameters, per-length short-text effects beyond Table 4.
