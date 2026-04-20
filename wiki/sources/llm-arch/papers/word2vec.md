# Efficient Estimation of Word Representations in Vector Space
- **Authors:** Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
- **Year:** 2013
- **URL:** https://arxiv.org/abs/1301.3781
- **Core Insight:** Words can be embedded as dense vectors where arithmetic works (king - man + woman = queen).
- **Guideline:** Represent discrete tokens as learned dense vectors (embeddings) rather than one-hot encodings. The resulting vector space captures semantic relationships and enables algebraic reasoning over meaning.
- **Relevant chapters:** Word embeddings, Tokenization, Representation learning, Foundations

## Abstract
We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.

## Key Contributions
- Proposed two efficient architectures for learning word vectors: Continuous Bag-of-Words (CBOW) and Skip-gram, both dramatically faster to train than previous neural language models
- Discovered that the learned vector space exhibits linear algebraic structure: vector("king") - vector("man") + vector("woman") approximates vector("queen"), demonstrating that embeddings capture relational semantics
- Showed that high-quality word vectors can be learned from 1.6 billion words in less than a day, making distributed representations practical at scale
- Created evaluation benchmarks for syntactic and semantic word relationships that became standard in the field
- Established that simple, shallow models trained on massive data can outperform deep models trained on less data for representation learning

## Why This Paper Matters
Word2Vec was the breakthrough that made the entire NLP community think in terms of dense vector representations. The embedding layer at the bottom of every Transformer -- including every LLM -- is a direct descendant of this work. The insight that meaning can be captured as geometry in vector space (and that vector arithmetic corresponds to semantic operations) remains foundational. Without Word2Vec demonstrating that embeddings work, the path to attention mechanisms and Transformers would have been far less clear.
