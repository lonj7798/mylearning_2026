<!-- scope: Dense Passage Retrieval — dual-encoder retrieval for open-domain QA
     deps: [[bert]], [[attention-is-all-you-need]]
     see-also: [[bert]]
-->

# Dense Passage Retrieval for Open-Domain Question Answering
- **Core Insight:** A simple dual-encoder trained with in-batch contrastive negatives produces dense passage embeddings that decisively outperform sparse BM25 retrieval for open-domain QA.
- **Guideline:** For retrieval in open-domain QA and RAG pipelines, train a BERT-based dual-encoder with contrastive objective and in-batch negatives rather than relying on lexical sparse retrievers.
- **Authors:** Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2004.04906
- **Relevant chapters:** Retrieval-augmented generation, open-domain QA, dense retrieval, long context vs. RAG tradeoffs

## Abstract
Open-domain question answering relies on efficient passage retrieval to select candidate contexts, where traditional sparse vector space models, such as TF-IDF or BM25, are the de facto method. In this work, we show that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual-encoder framework. When evaluated on a wide range of open-domain QA datasets, our dense retriever outperforms a strong Lucene-BM25 system greatly by 9%-19% absolute in terms of top-20 passage retrieval accuracy, and helps our end-to-end QA system establish new state-of-the-art on multiple open-domain QA benchmarks.

## Key Contributions
- Demonstrated that dense embeddings learned with a simple contrastive objective beat BM25 for passage retrieval — a result that was non-obvious at the time given the dominance of sparse lexical methods
- Introduced the two-tower (dual-encoder) BERT architecture as the canonical recipe for first-stage neural retrieval, decoupling question and passage encoding so passage embeddings can be pre-indexed
- Established in-batch negatives as an efficient and effective training signal, avoiding the need for explicit hard-negative mining in the basic setup while still achieving large gains
- Showed that a small amount of labeled QA data (tens of thousands of question-passage pairs) is sufficient to train a strong retriever — lexical matching was not a floor but a ceiling that learned representations clear easily
- Set new state-of-the-art on Natural Questions, TriviaQA, WebQuestions, CuratedTREC, and SQuAD open-domain benchmarks when paired with a downstream reader, establishing the retriever-reader pipeline as the reference architecture for open-domain QA and later RAG systems

## Key Figures/Tables to Study
- **Table 2** (top-k retrieval accuracy vs. BM25): DPR beats BM25 by 9-19% absolute at top-20 across NQ, TriviaQA, WebQuestions, CuratedTREC. This is the headline comparison that motivated the shift to dense retrieval.
- **Table 3** (in-batch negatives ablation): Shows how larger batch size increases the number of negatives per positive, directly improving retrieval accuracy — the core mechanism behind in-batch contrastive training.
- **Table 4** (gold vs. BM25 negatives vs. in-batch): Compares negative sampling strategies. In-batch with one BM25 hard negative per batch gives the strongest results.
- **Table 5** (cross-dataset generalization): DPR trained on one dataset transfers reasonably to others, showing the encoder learns general semantic matching rather than dataset-specific patterns.
- **Table 6** (end-to-end QA accuracy): DPR + reader sets new SOTA on NQ (41.5 EM), TriviaQA (56.8 EM), and WebQuestions (34.6 EM), closing the loop from retrieval gains to downstream QA gains.

## Architecture Details
- **Dual-encoder (two-tower) architecture:** Two independent BERT-base encoders — one for questions $E_Q(q)$, one for passages $E_P(p)$. Each takes the `[CLS]` token representation as the fixed-size embedding (768 dimensions).
- **Similarity function:** Dot product $\text{sim}(q, p) = E_Q(q)^\top E_P(p)$. Simple inner product chosen over cosine or L2 because it integrates cleanly with softmax and enables efficient maximum inner product search (MIPS) at inference.
- **Training objective:** Negative log-likelihood of the positive passage against a set of negatives, i.e., contrastive loss:
  $$L = -\log \frac{\exp(\text{sim}(q, p^+))}{\exp(\text{sim}(q, p^+)) + \sum_{j=1}^{n} \exp(\text{sim}(q, p_j^-))}$$
- **In-batch negatives:** For a batch of $B$ (question, positive-passage) pairs, each question uses the other $B-1$ positive passages in the batch as negatives. This gives $B-1$ free negatives per example with no extra forward passes. Typical batch size: 128, yielding 127 negatives per example.
- **Hard negatives:** One BM25-retrieved hard negative is added per example (a high-BM25-scoring passage that does not contain the answer). This forces the encoder to distinguish lexical overlap from semantic relevance.
- **Initialization:** Both encoders initialized from BERT-base-uncased (12 layers, 768 hidden, 12 heads, ~110M params per tower).
- **Inference:** Passage encoder is run once offline to build a FAISS index of all ~21M Wikipedia passages (100 tokens each). At query time, question encoder produces one embedding and retrieves top-k via approximate MIPS — a single forward pass plus near-constant-time index lookup.
- **Comparison vs. BM25:** BM25 scores documents by lexical term-frequency / inverse-document-frequency with length normalization. It cannot match a question like "Who invented the light bulb?" to a passage that says "Edison's incandescent filament" without shared surface tokens. DPR's learned embeddings do this naturally because BERT maps semantically related phrases into nearby vectors.
- **Key results:**
  - Top-20 passage retrieval accuracy on Natural Questions: **78.4% (DPR)** vs. 59.1% (BM25) — 19.3% absolute gain
  - Top-20 on TriviaQA: **79.4% (DPR)** vs. 66.9% (BM25) — 12.5% absolute gain
  - Top-100 on Natural Questions: **85.4% (DPR)** vs. 73.7% (BM25)
  - End-to-end exact match on Natural Questions: **41.5 (DPR+reader)** vs. 32.6 (previous SOTA, REALM)
- **Practical implications:** DPR became the default first-stage retriever for RAG systems. Later work (ANCE, Contriever, E5, BGE, Nomic, Cohere embed) refined the recipe with better negative mining, multi-stage training, and larger backbones, but the dual-encoder + contrastive + in-batch-negatives blueprint is unchanged.
- **Limitations:** Fixed-size bottleneck (single 768-d vector per passage) limits fine-grained matching; later methods like ColBERT introduced late-interaction to recover token-level granularity. DPR also needs in-domain training data — zero-shot transfer underperforms BM25 on some out-of-domain benchmarks (the finding that motivated Contriever and unsupervised dense retrievers).
