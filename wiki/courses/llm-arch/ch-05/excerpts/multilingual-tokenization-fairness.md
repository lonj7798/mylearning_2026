<!-- scope: multilingual fertility gap, fairness implications, vocabulary expansion strategies, parent: [[ch-05]] -->

# Multilingual Tokenization Fairness

Tokenization is the first place where language models encode bias. An English-centric tokenizer systematically disadvantages non-English users through higher fertility rates: more tokens per semantic unit means more compute, less context, and higher cost. This excerpt quantifies the problem, traces its structural causes, and surveys how LLaMA 3 and Qwen 3 addressed it.

---

## The Fertility Gap

**Fertility rate** = average number of tokens per word (or per semantic unit). Lower is better -- it means the tokenizer compresses more efficiently.

### Measured Fertility Rates (GPT-2 Tokenizer, 50,257 vocab)

| Language | Script | Fertility | Relative to English |
|----------|--------|-----------|-------------------|
| English | Latin | ~1.2 | 1.0x |
| Spanish | Latin | ~1.5 | 1.25x |
| German | Latin | ~1.6 | 1.33x |
| Russian | Cyrillic | ~2.2 | 1.83x |
| Arabic | Arabic | ~2.5 | 2.08x |
| Chinese | CJK | ~2.5 | 2.08x |
| Hindi | Devanagari | ~3.5 | 2.92x |
| Thai | Thai | ~4.0+ | 3.33x |
| Burmese | Myanmar | ~5.0+ | 4.17x |

---

## Why the Gap Exists

### Byte-Level BPE and UTF-8

Byte-level BPE (GPT-2 onward) operates on raw bytes. UTF-8 encoding uses variable byte widths:

- **ASCII (English, numbers, punctuation)**: 1 byte per character
- **Latin-extended (accented European)**: 2 bytes per character
- **CJK, Devanagari, Arabic**: 3 bytes per character
- **Emoji, rare scripts**: 4 bytes per character

A Chinese character is 3 bytes in UTF-8. If the BPE merge table has not merged those specific 3-byte sequences into a single token, that character consumes 3 tokens instead of 1. English "the" is 3 bytes but has been merged into a single token early in training because it is extremely frequent in the (English-dominated) corpus.

### Corpus Composition

BPE merges are frequency-driven. If the training corpus is 90% English, the most frequent byte pairs will be English patterns: " th", "he", " the", " in", "er", etc. These get merged early and consume few tokens. Non-English byte sequences, being less frequent, are merged later or not at all, resulting in longer token sequences.

---

## Downstream Consequences

### 1. Context Window Consumption

A model with a 4,096-token context window can fit approximately:
- **English**: ~3,400 words (at 1.2 tokens/word)
- **Hindi**: ~1,170 words (at 3.5 tokens/word)

The Hindi user gets **3x less content** in the same context window. For tasks that require long-context reasoning (document summarization, multi-turn dialogue), this is a significant capability gap.

### 2. Compute Cost

Self-attention cost is $O(N^2)$ in sequence length $N$. If the same semantic content is 3x longer in Hindi than English, the attention cost is ~9x higher. Even with linear-complexity attention variants ([[ch-16]]), the cost scales linearly with sequence length -- 3x more tokens means 3x more compute.

### 3. API Cost

Providers that charge per token (OpenAI, Anthropic, Google) pass the fertility gap directly to users. A Hindi user pays ~3x more than an English user for equivalent semantic content. This is a structural economic disadvantage, not a model capability issue.

### 4. Training Data Efficiency

During training, the model processes a fixed number of tokens per batch. If Hindi text is 3x more tokens per semantic unit, the model sees 3x less Hindi semantic content per training step. This compounds the existing underrepresentation of non-English text in training corpora.

---

## How LLaMA 3 and Qwen 3 Addressed It

### LLaMA 3: 32K to 128K Vocabulary

LLaMA 3 ([[llama-3|report]]) quadrupled its vocabulary from 32,000 to 128,256 tokens. The expansion specifically targeted:
- Common subwords in top-30 non-English languages
- CJK character coverage (most common characters became single tokens)
- Code tokens (programming language keywords and patterns)

**Results**: 15-20% reduction in English token count, 30-50% reduction for non-English languages. At 405B scale, the 128K embedding table is only 0.26% of total parameters -- the cost is negligible.

### Qwen 3: 151K Vocabulary, 119 Languages

Qwen 3 ([[qwen-3|report]]) pushed further to 151,669 tokens covering 119 languages. The vocabulary was designed jointly with the training data mix to ensure that every vocabulary entry receives adequate training signal. Key design choices:
- Tied embeddings for small models (0.6B, 1.7B) to manage the parameter cost
- Untied embeddings for 8B+ where the vocabulary cost is proportionally smaller
- Explicit fertility rate targets per language family

---

## The Fundamental Tradeoff

Expanding vocabulary improves non-English compression but has costs:

| Factor | Larger Vocab | Smaller Vocab |
|--------|-------------|---------------|
| Fertility (non-English) | Lower (better) | Higher (worse) |
| Embedding parameters | More (costly for small models) | Fewer |
| Softmax output cost | More per step | Less per step |
| Token diversity | Higher (more distinct tokens to learn) | Lower |
| Training data coverage | Needs multilingual data for all entries | Can focus on dominant language |

The optimal vocabulary size depends on the target model scale and language distribution. For English-only models, 32K is sufficient. For multilingual models serving diverse populations, 128K-150K is the emerging standard.

---

## Measuring Fairness

A principled fairness metric for tokenization:

$$\text{Fairness ratio}(L) = \frac{\text{fertility}(L)}{\text{fertility}(\text{English})}$$

If this ratio exceeds 2.0 for any target language $L$, the tokenizer needs rebalancing. The goal is not equal fertility (impossible due to script differences) but bounded disparity.

---

## References

- [[llama-3|Meta AI "The Llama 3 Herd of Models" (2024) (report)]]
- [[qwen-3|Qwen Team "Qwen3 Technical Report" (2025) (report)]]
- [[gpt-2|Radford et al. "Language Models are Unsupervised Multitask Learners" (2019) (paper)]]
- Petrov et al. "Language Model Tokenizers Introduce Unfairness Between Languages" (2023)
