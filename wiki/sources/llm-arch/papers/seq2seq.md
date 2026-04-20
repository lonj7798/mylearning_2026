# Sequence to Sequence Learning with Neural Networks
- **Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- **Year:** 2014
- **URL:** https://arxiv.org/abs/1409.3215
- **Core Insight:** Encoder-decoder with fixed-length vector can translate -- but the bottleneck motivates attention.
- **Guideline:** The encoder-decoder pattern is powerful for mapping one sequence to another, but always consider whether a fixed-length bottleneck will limit performance on longer inputs.
- **Relevant chapters:** Encoder-decoder architectures, Sequence-to-sequence models, RNN foundations

## Abstract
Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT'14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM's performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.

## Key Contributions
- Established the encoder-decoder paradigm: one LSTM encodes the source sequence into a fixed-length vector, another LSTM decodes the target sequence from that vector
- Demonstrated that deep LSTMs (4 layers) significantly outperform shallow ones, showing depth matters for sequence modeling
- Discovered the surprising trick that reversing source sentence order improves translation quality by creating shorter-range dependencies
- Showed that learned representations capture meaningful semantic structure, with similar sentences clustering together in vector space
- Proved that end-to-end neural approaches could compete with heavily engineered statistical machine translation systems

## Why This Paper Matters
This paper established the encoder-decoder framework that became the dominant paradigm for sequence-to-sequence tasks, from translation to summarization to dialogue. Its very success also revealed the key limitation -- compressing an entire input sequence into a single fixed-length vector -- that directly motivated Bahdanau's attention mechanism. The seq2seq architecture remains the conceptual ancestor of the Transformer's encoder-decoder structure.
