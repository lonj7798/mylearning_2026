<!-- scope: Retrieval-Augmented Generation — combine dense retrieval with seq2seq generation for knowledge-intensive NLP
     deps: [[attention-is-all-you-need]], [[bert]]
     see-also: [[gpt-3]], [[bahdanau-attention]]
-->

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Core Insight:** Pairing a parametric seq2seq generator with a non-parametric dense retriever over an external corpus beats parametric-only models on knowledge-intensive tasks and lets the knowledge store be swapped without retraining.
- **Guideline:** For factual or knowledge-heavy tasks, retrieve top-k passages with a dense (DPR-style) encoder and condition a pretrained seq2seq generator (e.g., BART) on them — jointly fine-tune the query encoder and generator while keeping the document encoder frozen.
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
- **Year:** 2020
- **URL:** https://arxiv.org/abs/2005.11401
- **Relevant chapters:** Long context, retrieval augmentation, knowledge-intensive NLP, open-domain QA

## Abstract
Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, and another which can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state of the art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

## Key Contributions
- Introduced a general-purpose fine-tuning recipe that hybridizes parametric (seq2seq) and non-parametric (dense index) memory, applicable to any knowledge-intensive NLP task without task-specific architectures
- Formulated two retrieval-conditioning variants: **RAG-Sequence** (one retrieved document conditions the entire output) and **RAG-Token** (each output token can attend to a different retrieved document), trading off coherence vs. per-token flexibility
- Set state-of-the-art on three open-domain QA benchmarks (Natural Questions, TriviaQA, WebQuestions), beating both closed-book parametric models and task-specific extractive retrieve-and-read pipelines
- Demonstrated that the non-parametric memory can be **hot-swapped at inference time** — replacing the Wikipedia index updates the model's world knowledge without any retraining, directly addressing staleness of parametric models
- Showed that RAG generates more specific, diverse, and factually grounded text than a BART baseline on open-domain generation, with provenance traceable back to retrieved passages

## Key Figures/Tables to Study
- **Figure 1** (RAG architecture): The canonical diagram showing DPR query encoder, MIPS retrieval over the document index, and BART generator conditioned on retrieved passages. Study which components are trained and which are frozen.
- **Table 1** (Open-domain QA results): RAG vs. closed-book T5/BART and vs. extractive retrieve-and-read systems (REALM, DPR). Note RAG's gains on Natural Questions, TriviaQA, WebQuestions, and CuratedTREC.
- **Table 2** (Abstractive QA on MS-MARCO): RAG outperforms BART despite BART having access to gold passages — demonstrates that retrieved passages plus generation beats generation alone, even with oracle context.
- **Table 3** (Jeopardy question generation): Human evaluation finds RAG-generated questions more factual and specific than BART. Shows benefit on generation tasks, not just QA.
- **Table 6** (Index hot-swap experiment): Swapping a 2016 Wikipedia index for a 2018 index updates answers to time-sensitive questions with **no retraining**. This is the clearest demonstration of non-parametric memory's advantage.

## Architecture Details
- **Retriever (DPR):** Dense Passage Retriever with two BERT-base encoders. Query encoder $q(x) = \text{BERT}_q(x)$ produces a dense query vector; document encoder $d(z) = \text{BERT}_d(z)$ produces a dense document vector. Retrieval score is the inner product $q(x)^T d(z)$.
- **Document index:** 21M 100-word chunks from Wikipedia (Dec 2018 dump), each encoded once with the frozen document encoder. MIPS (Maximum Inner Product Search) via FAISS retrieves the top-k documents (typically k=5 or k=10) in sub-linear time.
- **Generator:** BART-large (400M parameters), a pretrained denoising seq2seq transformer. Input is the concatenation of the query $x$ and a retrieved document $z$; output is the target sequence $y$.
- **RAG-Sequence:** Marginalizes over retrieved documents at the **sequence level**: $p(y|x) = \sum_{z \in \text{top-k}} p_\eta(z|x) \prod_i p_\theta(y_i | x, z, y_{<i})$. A single document conditions the entire generation, so outputs are document-coherent but less flexible.
- **RAG-Token:** Marginalizes over retrieved documents at the **token level**: $p(y|x) = \prod_i \sum_{z \in \text{top-k}} p_\eta(z|x) p_\theta(y_i | x, z, y_{<i})$. Each output token can effectively draw from a different document, enabling synthesis across passages at the cost of coherence.
- **Training:** Jointly fine-tune the query encoder $\text{BERT}_q$ and generator BART with a standard cross-entropy objective on $(x, y)$ pairs. **The document encoder $\text{BERT}_d$ and the document index are frozen** — re-encoding 21M passages every update would be prohibitively expensive. Gradient flows to the retriever only through the query encoder and the softmax over the top-k document scores.
- **Top-k marginalization:** Because exact marginalization over all 21M documents is intractable, RAG approximates the sum by restricting to the top-k retrieved documents per query. This is the core approximation that makes end-to-end differentiable training feasible.
- **Decoding:** RAG-Token uses standard beam search because the per-token marginal can be computed on-the-fly. RAG-Sequence requires a "thorough" decoding procedure that runs beam search per document and then re-ranks, since different documents yield different hypothesis sets.
- **Open-domain QA results:** Natural Questions 44.5 EM (vs. 40.4 for REALM, 36.6 for T5-11B closed-book), TriviaQA 56.8 EM, WebQuestions 45.2 EM, CuratedTREC 52.2 EM — state-of-the-art at publication on all four.
- **Abstractive generation (MS-MARCO):** RAG-Token Bleu-1 of 43.5 vs. BART's 41.6, with human raters rating RAG outputs as more specific and factual.
- **Jeopardy generation:** RAG-Token generates questions rated more factual (42.7% vs. 7.1%) and more specific (37.4% vs. 16.8%) than BART by human evaluators.
- **Non-parametric memory advantage:** Swapping the Wikipedia index from Dec 2016 to Dec 2018 corrects answers to questions about time-sensitive facts (e.g., heads of state) — with **zero parameter updates**. This cleanly separates "how to generate" (parametric) from "what to know" (non-parametric).
- **Limitations:** Retrieval quality bounds generation quality — if top-k misses the relevant passage, the generator cannot recover. The frozen document encoder means the index cannot adapt to the fine-tuning distribution. Compute cost scales with k (the generator is conditioned on k separate (x, z) inputs).
