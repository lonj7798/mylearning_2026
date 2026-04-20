<!-- scope: BERT — bidirectional pre-training via masked language modeling
     deps: [[attention-is-all-you-need]]
     see-also: [[gpt-1]], [[gpt-2]]
-->

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Core Insight:** Bidirectional context from masking produces richer representations than left-to-right language modeling.
- **Guideline:** For understanding tasks (classification, extraction, retrieval), prefer an encoder that sees full context; reserve autoregressive decoders for generation.
- **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- **Year:** 2018
- **URL:** https://arxiv.org/abs/1810.04805
- **Relevant chapters:** Bidirectional pre-training, masked language modeling, encoder-only Transformers, fine-tuning, sentence-pair tasks

## Abstract
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

## Key Contributions
- Introduced deep bidirectional pre-training using Masked Language Modeling (MLM), where random tokens are masked and the model must predict them using both left and right context -- a fundamental departure from left-to-right language modeling
- Demonstrated that pre-trained bidirectional representations dramatically outperform unidirectional (GPT) and shallow bidirectional (ELMo) approaches on a wide range of NLP tasks
- Introduced the Next Sentence Prediction (NSP) pre-training objective to capture inter-sentence relationships, enabling tasks like question answering and natural language inference
- Showed that a single architecture with minimal task-specific modifications (just one output layer) can achieve state-of-the-art on 11 diverse NLP tasks simultaneously
- Established the encoder-only Transformer as the dominant architecture for discriminative NLP tasks, complementing the decoder-only approach of GPT

## Key Figures/Tables to Study
- **Figure 1** (BERT vs. GPT vs. ELMo): The canonical comparison figure showing three pre-training strategies. BERT uses bidirectional context in all layers; GPT uses left-to-right; ELMo concatenates separate left-to-right and right-to-left models. Study this to understand why BERT's approach yields richer representations.
- **Figure 3** (Fine-tuning on different tasks): Shows how the same pre-trained BERT architecture is adapted for sentence classification, sentence-pair classification, question answering, and tagging tasks with minimal modifications.
- **Table 1** (GLUE benchmark results): BERT-Large achieves 80.5% GLUE score, a 7.7-point improvement over prior SOTA. Compare BERT-Base and BERT-Large to understand the effect of scale.
- **Table 2** (SQuAD results): Strong gains on both extractive QA tasks, showing the power of bidirectional context for reading comprehension.
- **Table 5** (Ablation over pre-training objectives): Critical table showing the impact of removing NSP and comparing MLM to left-to-right LM. Proves that bidirectionality is the key ingredient.

## Architecture Details
- **Architecture:** Encoder-only Transformer (uses only the encoder stack from the original Transformer)
- **BERT-Base:** 12 layers, 768 hidden, 12 attention heads, 110M parameters
- **BERT-Large:** 24 layers, 1024 hidden, 16 attention heads, 340M parameters
- **Context window:** 512 tokens
- **Pre-training objectives:**
  - Masked Language Modeling (MLM): 15% of tokens randomly selected; of those, 80% replaced with [MASK], 10% replaced with random token, 10% kept unchanged
  - Next Sentence Prediction (NSP): Binary classification of whether sentence B follows sentence A
- **Special tokens:** [CLS] (classification), [SEP] (separator), [MASK]
- **Input representation:** Token embeddings + segment embeddings (sentence A vs. B) + position embeddings, all summed
- **Pre-training data:** BooksCorpus (800M words) + English Wikipedia (2,500M words)
- **Training hardware:** 4 Cloud TPUs for BERT-Base (4 days), 16 Cloud TPUs for BERT-Large (4 days)
- **Optimizer:** Adam with learning rate 1e-4, beta_1=0.9, beta_2=0.999, L2 weight decay of 0.01
- **Batch size:** 256 sequences
- **Training steps:** 1,000,000
- **Tokenization:** WordPiece with 30,000 token vocabulary
- **Activation function:** GELU
- **Dropout:** 0.1 on all layers
