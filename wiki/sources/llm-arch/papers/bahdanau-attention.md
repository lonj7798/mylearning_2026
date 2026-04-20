# Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year:** 2014
- **URL:** https://arxiv.org/abs/1409.0473
- **Core Insight:** Attention as soft alignment lets models focus on relevant parts of input, eliminating the fixed-length bottleneck.
- **Guideline:** When your model must compress variable-length input into a fixed representation, add an attention mechanism so it can selectively focus on the most relevant parts at each decoding step.
- **Relevant chapters:** Attention mechanisms, Encoder-decoder architectures, Sequence-to-sequence models

## Abstract
Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition.

## Key Contributions
- Introduced the attention mechanism for neural machine translation, allowing the decoder to look back at all encoder hidden states rather than relying on a single fixed-length context vector
- Demonstrated that soft alignment (learned attention weights) naturally discovers word-level correspondences between source and target languages
- Showed that attention eliminates the information bottleneck of compressing an entire sentence into one vector, dramatically improving performance on longer sentences
- Provided interpretable alignment visualizations showing the model learns linguistically meaningful correspondences
- Established the foundation for all subsequent attention-based architectures, including the Transformer

## Why This Paper Matters
This paper is the origin of the attention mechanism that would later become the sole building block of the Transformer architecture. By replacing the fixed-length bottleneck with a dynamic, content-based addressing scheme, Bahdanau et al. solved a fundamental limitation of encoder-decoder models and opened the door to architectures that scale gracefully with sequence length. Without this insight, the "Attention Is All You Need" paper would not have been possible.
