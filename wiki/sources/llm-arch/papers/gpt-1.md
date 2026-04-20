<!-- scope: first GPT — generative pre-training + discriminative fine-tuning
     deps: [[ch-01]], [[attention-is-all-you-need]]
     see-also: [[gpt-2]], [[bert]]
-->

# Improving Language Understanding by Generative Pre-Training
- **Core Insight:** Generative pre-training on unlabeled text followed by discriminative fine-tuning transfers effectively across diverse NLP tasks.
- **Guideline:** Pre-train generatively first, then fine-tune; the two-stage recipe beats task-specific architectures.
- **Authors:** Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever
- **Year:** 2018
- **URL:** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- **Relevant chapters:** Generative pre-training, transfer learning, fine-tuning paradigm, decoder-only Transformer

## Abstract
Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately. We demonstrate that large gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture. We evaluate on a wide range of benchmarks for natural language understanding and show that our general task-agnostic model outperforms discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon the state of the art in 9 out of the 12 tasks studied. For instance, we achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE), and 1.5% on textual entailment (MultiNLI).

## Key Contributions
- Established the two-stage paradigm of unsupervised pre-training followed by supervised fine-tuning, which became the dominant NLP methodology
- Showed that a single decoder-only Transformer language model can be adapted to a wide range of downstream tasks with minimal architectural changes
- Introduced task-aware input transformations (structured token sequences with delimiters) that allow a single model architecture to handle classification, entailment, similarity, and QA tasks
- Demonstrated that generative pre-training on BooksCorpus (7,000+ unpublished books) produces representations that transfer broadly to diverse NLP benchmarks
- Achieved state-of-the-art results on 9 of 12 NLP benchmarks, proving the effectiveness of unsupervised pre-training over task-specific architectures

## Key Figures/Tables to Study
- **Figure 1** (left: Transformer architecture, right: input transformations): Shows the 12-layer decoder-only Transformer and -- crucially -- how different task types (classification, entailment, similarity, multiple choice) are formatted as linear token sequences. This figure defines the fine-tuning paradigm used by all subsequent GPT models.
- **Table 1** (Results on NLI tasks): Performance across multiple natural language inference benchmarks, demonstrating broad transferability.
- **Table 2** (Results across all 12 tasks): The headline result table. Compare against task-specific baselines to appreciate the generality of the approach.
- **Table 5** (Ablation analysis): Shows the contribution of each component -- pre-training, auxiliary LM objective during fine-tuning, and the Transformer vs. LSTM architecture.

## Architecture Details
- **Architecture:** Decoder-only Transformer (no encoder)
- **Number of layers:** 12
- **Model dimension (d_model):** 768
- **Number of attention heads:** 12
- **Feed-forward dimension:** 3072 (4x model dimension)
- **Context window:** 512 tokens
- **Total parameters:** ~117M
- **Pre-training corpus:** BooksCorpus (~7,000 unique unpublished books, ~800M words)
- **Pre-training objective:** Standard language modeling (next-token prediction)
- **Fine-tuning objective:** Supervised task loss + auxiliary language model loss (weighted by lambda=0.5)
- **Activation function:** GELU (Gaussian Error Linear Unit)
- **Positional embeddings:** Learned (not sinusoidal)
- **Optimizer:** Adam with max learning rate 2.5e-4
- **Learning rate schedule:** Linear warmup over first 2000 updates, then cosine annealing to 0
- **Tokenization:** BPE (Byte Pair Encoding) with ~40,000 merges
- **Batch size:** 64 sequences of 512 tokens
- **Training duration:** 100 epochs on BooksCorpus
- **Regularization:** Dropout (0.1), L2 weight decay (0.01)
