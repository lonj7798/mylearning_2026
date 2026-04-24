# Chapter 5: Tokenization

<!-- scope: BPE, WordPiece, SentencePiece, vocabulary size tradeoffs, tokenization failures, architecture impact
     deps: [[ch-01]]
     see-also: [[ch-06]], [[ch-07]]
-->

## Companion Materials

**Interactive Figures:**
- [BPE Merge Animation](figures/bpe-merge-animation.html) -- step through the BPE algorithm on a live corpus; watch pairs get counted, merged, and added to the vocabulary
- [Vocab Size vs Sequence Length Tradeoff](figures/vocab-size-tradeoff.html) -- interactive explorer showing how vocabulary size affects fertility, embedding cost, and attention compute across real models (BERT, GPT-2, LLaMA 1/3, Qwen 3)

**Deep-Dive Excerpts:**
- [BPE Algorithm Step-by-Step Walkthrough](excerpts/bpe-algorithm-walkthrough.md) -- full worked example from corpus to vocabulary, byte-level BPE, merge table as compressed knowledge
- [SolidGoldMagikarp Deep-Dive](excerpts/solidgoldmagikarp-deep-dive.md) -- root cause analysis of glitch tokens, the tokenizer-model corpus mismatch, mitigation strategies
- [Multilingual Tokenization Fairness](excerpts/multilingual-tokenization-fairness.md) -- fertility gap quantification, downstream consequences (context, compute, cost), how LLaMA 3 and Qwen 3 addressed it

---

## Overview

Tokenization is the interface between raw text and the model's numerical world. Every decision made here propagates through the entire architecture: the embedding table size, the effective sequence length, the compute cost of attention, and the model's ability to handle multilingual text, code, and edge cases. It is easy to treat tokenization as a preprocessing detail. That is a mistake — vocabulary design is an architectural decision with first-order consequences for model quality, efficiency, and fairness.

The core tension is compression efficiency versus vocabulary cost. A larger vocabulary compresses text into fewer tokens (shorter sequences, cheaper attention), but each additional vocabulary entry adds a row to the embedding table and the output projection, increasing parameter count and memory. Every frontier model navigates this tradeoff differently: [[gpt-2|GPT-2 (paper)]] chose 50,257 tokens, [[bert|BERT (paper)]] chose 30,522, [[llama-1|LLaMA 1 (report)]] chose 32,000, [[llama-3|LLaMA 3 (report)]] quadrupled to 128,256, and [[qwen-3|Qwen 3 (report)]] pushed to 151,669. These are not arbitrary numbers — each reflects a deliberate bet on the language distribution the model will serve.

This chapter covers the three dominant tokenization algorithms (BPE, WordPiece, SentencePiece/Unigram), the architectural consequences of vocabulary size, and the failure modes that arise when tokenization breaks down — including the infamous SolidGoldMagikarp phenomenon that reveals how vocabulary artifacts can cause undefined model behavior.

---

## 1. Byte Pair Encoding: The Dominant Algorithm

BPE was originally a data compression algorithm (Gage, 1994), adapted for neural machine translation by Sennrich et al. (2016) and adopted by the GPT lineage. It builds a vocabulary bottom-up by iteratively merging the most frequent adjacent symbol pairs.

### 1.1 The Algorithm

Starting from a base vocabulary of individual characters (or bytes), BPE proceeds:

1. **Initialize:** Split all training text into individual characters. Append a special end-of-word marker (e.g., `</w>`) to distinguish word boundaries.
2. **Count:** Find the most frequent adjacent pair of symbols across the entire corpus.
3. **Merge:** Replace every occurrence of that pair with a new single symbol. Add the new symbol to the vocabulary.
4. **Repeat:** Go to step 2. Continue until the vocabulary reaches the desired size or no pair exceeds a frequency threshold.

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0; font-family:monospace;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">BPE Merge Walkthrough: "low lower lowest"</div>
<div style="display:flex; flex-direction:column; gap:12px;">

<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">Step 0 — Base characters</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:6px;">l o w &lt;/w&gt; &nbsp; l o w e r &lt;/w&gt; &nbsp; l o w e s t &lt;/w&gt;</div>
<div style="color:#888; font-size:11px; margin-top:4px;">Vocab: {l, o, w, e, r, s, t, &lt;/w&gt;} — 8 symbols</div>
</div>

<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">Merge 1 — Most frequent pair: (l, o) appears 3x</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:6px;"><span style="color:#e94560; font-weight:bold;">lo</span> w &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">lo</span> w e r &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">lo</span> w e s t &lt;/w&gt;</div>
<div style="color:#888; font-size:11px; margin-top:4px;">Vocab: {l, o, w, e, r, s, t, &lt;/w&gt;, <span style="color:#e94560;">lo</span>} — 9 symbols</div>
</div>

<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">Merge 2 — Most frequent pair: (lo, w) appears 3x</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:6px;"><span style="color:#e94560; font-weight:bold;">low</span> &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">low</span> e r &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">low</span> e s t &lt;/w&gt;</div>
<div style="color:#888; font-size:11px; margin-top:4px;">Vocab: {..., <span style="color:#e94560;">low</span>} — 10 symbols</div>
</div>

<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">Merge 3 — Most frequent pair: (e, r) appears 1x, (e, s) appears 1x, (low, &lt;/w&gt;) appears 1x — tie broken arbitrarily, say (low, e) at 2x</div>
<div style="color:#e0e0e0; font-size:13px; margin-top:6px;">low &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">lowe</span> r &lt;/w&gt; &nbsp; <span style="color:#e94560; font-weight:bold;">lowe</span> s t &lt;/w&gt;</div>
<div style="color:#888; font-size:11px; margin-top:4px;">Vocab: {..., <span style="color:#e94560;">lowe</span>} — 11 symbols</div>
</div>

<div style="background:#16213e; border-radius:8px; padding:12px;">
<div style="color:#e94560; font-size:12px; font-weight:bold;">Merge 4 — (lower, &lt;/w&gt;) or (lowe, r) or (lowe, s)... continues until target vocab size</div>
<div style="color:#888; font-size:11px; margin-top:4px;">Each merge creates a new token that captures a recurring substring pattern.</div>
</div>

</div>
<div style="color:#888; font-size:12px; margin-top:16px; font-family:sans-serif;">
The merge list is ordered and deterministic. At inference time, the same merges are applied in the same order to tokenize new text. Common words eventually become single tokens; rare words decompose into subword pieces.
</div>
</div>

**Critical property:** BPE merges are greedy and frequency-driven. The algorithm has no notion of morphology or meaning — it discovers subword units purely from co-occurrence statistics. This means BPE often creates linguistically arbitrary splits ("unhappiness" might tokenize as "un" + "happiness" or "unh" + "app" + "iness" depending on corpus statistics). The algorithm is indifferent.

### 1.2 Byte-Level BPE (GPT-2 Onward)

Original BPE operates on Unicode characters, which means the base vocabulary must include every Unicode codepoint the model might encounter — or map unknown characters to a special `<UNK>` token. GPT-2 solved this by operating on **raw bytes** instead of characters.

The base vocabulary is exactly 256 entries (one per byte value). Any Unicode character decomposes into 1-4 UTF-8 bytes, so any text is representable. No `<UNK>` token is needed. GPT-2's final vocabulary of 50,257 tokens = 256 byte tokens + 50,000 BPE merges + 1 special end-of-text token.

The tradeoff: rare Unicode characters (e.g., Chinese characters, emoji) may require multiple byte tokens, inflating sequence length for non-Latin scripts. This is the root cause of multilingual tokenization unfairness — a point we return to in Section 4.

---

## 2. WordPiece and SentencePiece: The Alternatives

### 2.1 WordPiece (BERT)

WordPiece, developed at Google for Japanese/Korean segmentation and adopted by BERT, resembles BPE but differs in its merge criterion. Where BPE merges the most **frequent** pair, WordPiece merges the pair that **maximizes the likelihood** of the training corpus under a unigram language model.

Concretely, WordPiece scores each candidate merge (A, B) $\rightarrow$ AB as:

$$\text{score}(A, B) = \frac{\text{freq}(AB)}{\text{freq}(A) \times \text{freq}(B)}$$

This is essentially pointwise mutual information. It favors merges where A and B co-occur far more than chance would predict, even if the raw frequency is low. BPE would merge "th" + "e" early (high frequency); WordPiece might prefer merging rarer but more predictable pairs first.

BERT's WordPiece vocabulary has 30,522 tokens. Subword tokens that continue a word are prefixed with `##` (e.g., "playing" $\rightarrow$ "play" + "##ing"). This explicit continuation marker is cosmetically different from BPE's approach but functionally similar — both signal that a token is not a word boundary.

### 2.2 SentencePiece and the Unigram Model

SentencePiece (Kudo & Richardson, 2018) addresses a practical limitation of both BPE and WordPiece: they require pre-tokenized (whitespace-split) input. This works for English but fails for languages without explicit word boundaries — Chinese, Japanese, Thai. SentencePiece treats the input as a raw character stream, including whitespace (represented as `_` or `\u2581`), and learns segmentation end-to-end.

SentencePiece supports two algorithms:
- **BPE mode:** Standard BPE on the raw character stream.
- **Unigram mode:** Starts with a very large vocabulary and *prunes* it down, the inverse of BPE's bottom-up approach.

The **unigram language model** approach is the more interesting one. It begins with a large candidate vocabulary (e.g., all substrings up to some length) and iteratively removes tokens that least decrease the corpus log-likelihood. At each step, it computes the marginal loss of removing each token and drops the lowest-impact ones. The final vocabulary is the set of subwords that best compress the corpus under a unigram assumption (each token is generated independently).

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Tokenization Algorithm Comparison</div>
<div style="overflow-x:auto;">
<table style="width:100%; border-collapse:collapse; font-size:13px; color:#e0e0e0;">
<tr style="border-bottom:2px solid #e94560;">
<th style="padding:10px; text-align:left; color:#e94560;">Property</th>
<th style="padding:10px; text-align:center; color:#e94560;">BPE</th>
<th style="padding:10px; text-align:center; color:#e94560;">WordPiece</th>
<th style="padding:10px; text-align:center; color:#e94560;">Unigram (SentencePiece)</th>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:10px;">Build direction</td>
<td style="padding:10px; text-align:center;">Bottom-up (merge)</td>
<td style="padding:10px; text-align:center;">Bottom-up (merge)</td>
<td style="padding:10px; text-align:center;">Top-down (prune)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:10px;">Merge criterion</td>
<td style="padding:10px; text-align:center;">Frequency</td>
<td style="padding:10px; text-align:center;">Likelihood (PMI)</td>
<td style="padding:10px; text-align:center;">Marginal likelihood loss</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:10px;">Segmentation</td>
<td style="padding:10px; text-align:center;">Deterministic</td>
<td style="padding:10px; text-align:center;">Deterministic</td>
<td style="padding:10px; text-align:center;">Probabilistic (can sample)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:10px;">Requires pre-tokenization</td>
<td style="padding:10px; text-align:center;">Yes (whitespace)</td>
<td style="padding:10px; text-align:center;">Yes (whitespace)</td>
<td style="padding:10px; text-align:center;">No (raw stream)</td>
</tr>
<tr style="border-bottom:1px solid #333;">
<td style="padding:10px;">Notable users</td>
<td style="padding:10px; text-align:center;">GPT-2/3/4, LLaMA</td>
<td style="padding:10px; text-align:center;">BERT, DistilBERT</td>
<td style="padding:10px; text-align:center;">T5, ALBERT, XLNet, Qwen</td>
</tr>
<tr>
<td style="padding:10px;">Handles CJK natively</td>
<td style="padding:10px; text-align:center;">Via byte fallback</td>
<td style="padding:10px; text-align:center;">No (needs pre-seg)</td>
<td style="padding:10px; text-align:center;">Yes</td>
</tr>
</table>
</div>
</div>

**A key advantage of the unigram model:** It assigns a probability to each segmentation, which means the same string can have multiple valid tokenizations with different probabilities. During training, you can sample different segmentations as a form of data augmentation (subword regularization). BPE always produces the same segmentation — it is deterministic by construction.

---

## 3. Vocabulary Size: The Central Architectural Tradeoff

Vocabulary size is not a hyperparameter you tune — it is a design constant that shapes the entire model. The consequences flow in two directions: **upstream** into the embedding table (parameters and memory) and **downstream** into sequence length (attention cost and positional encoding demands).

### 3.1 Embedding Table Cost

The embedding table maps each vocabulary token to a dense vector of dimension $d_\text{model}$. The output projection (or language modeling head) maps $d_\text{model}$ back to vocabulary logits. In models that tie input and output embeddings (GPT-2, LLaMA), the cost is:

$$\text{Embedding parameters} = V \times d_\text{model}$$

In models that untie them (LLaMA 3 8B), the cost doubles:

$$\text{Embedding parameters} = 2 \times V \times d_\text{model}$$

<div style="background:#1a1a2e; border-radius:12px; padding:24px; margin:20px 0;">
<div style="color:#e0e0e0; font-size:14px; margin-bottom:16px; font-family:sans-serif; font-weight:bold;">Embedding Table Size Across Models (FP16 = 2 bytes/param)</div>
<div style="display:flex; flex-direction:column; gap:8px;">

<div style="display:flex; align-items:center; gap:12px;">
<div style="color:#aaa; font-size:12px; width:120px; text-align:right;">BERT</div>
<div style="background:#0f3460; height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; width:80px;">
<span style="color:#e0e0e0; font-size:11px;">30,522</span>
</div>
<div style="color:#888; font-size:11px;">x 768 = 23.4M params (44.9 MB)</div>
</div>

<div style="display:flex; align-items:center; gap:12px;">
<div style="color:#aaa; font-size:12px; width:120px; text-align:right;">GPT-2 XL</div>
<div style="background:#0f3460; height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; width:110px;">
<span style="color:#e0e0e0; font-size:11px;">50,257</span>
</div>
<div style="color:#888; font-size:11px;">x 1,600 = 80.4M params (153.8 MB)</div>
</div>

<div style="display:flex; align-items:center; gap:12px;">
<div style="color:#aaa; font-size:12px; width:120px; text-align:right;">LLaMA 1</div>
<div style="background:#0f3460; height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; width:88px;">
<span style="color:#e0e0e0; font-size:11px;">32,000</span>
</div>
<div style="color:#888; font-size:11px;">x 4,096 = 131M params (250 MB) [tied]</div>
</div>

<div style="display:flex; align-items:center; gap:12px;">
<div style="color:#e94560; font-size:12px; width:120px; text-align:right; font-weight:bold;">LLaMA 3 8B</div>
<div style="background:#e94560; height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; width:340px;">
<span style="color:#fff; font-size:11px;">128,256</span>
</div>
<div style="color:#888; font-size:11px;">x 4,096 x 2 = 1.05B params (2.0 GB) [untied]</div>
</div>

<div style="display:flex; align-items:center; gap:12px;">
<div style="color:#e94560; font-size:12px; width:120px; text-align:right; font-weight:bold;">Qwen 3</div>
<div style="background:#e94560; height:24px; border-radius:4px; display:flex; align-items:center; padding:0 8px; width:400px;">
<span style="color:#fff; font-size:11px;">151,669</span>
</div>
<div style="color:#888; font-size:11px;">x 4,096 = 621M params (1.2 GB) [tied for small models]</div>
</div>

</div>
<div style="color:#888; font-size:12px; margin-top:16px; font-family:sans-serif;">
LLaMA 3's untied embeddings at 128K vocab consume 1.05B parameters — 13% of the 8B model's total. At 405B scale, the ratio drops to 0.26%, making the cost negligible. Large vocabularies are proportionally more expensive for small models.
</div>
</div>

The key insight: **embedding table cost is a fixed overhead that scales with vocab size, not with model depth.** For large models (70B+), a 128K vocabulary is a rounding error. For small models (0.6B-8B), it can be a significant fraction of total parameters. This is why Qwen 3 ties embeddings for its 0.6B and 1.7B models but unties them for 8B+.

### 3.2 Sequence Length: The Downstream Consequence

A larger vocabulary compresses text into fewer tokens. Fewer tokens means shorter sequences, and shorter sequences mean quadratically less attention compute (or linearly less with linear attention, [[ch-16]]). This is the other side of the tradeoff: what you pay in embedding parameters, you save in attention FLOPs.

The **fertility rate** measures this: the average number of tokens per word. Lower is better — it means the tokenizer compresses more aggressively.

Approximate fertility rates for English text:
- LLaMA 1 (32K vocab): ~1.3 tokens/word
- GPT-2 (50K vocab): ~1.2 tokens/word
- LLaMA 3 (128K vocab): ~1.0-1.1 tokens/word

The LLaMA 3 technical report specifically notes that the 4x vocabulary expansion from 32K to 128K significantly reduces token counts per input, improving both training efficiency (more text per training sequence) and inference latency (fewer autoregressive steps).

### 3.3 The Multilingual Dimension

Fertility rate varies dramatically across languages, and this is where vocabulary design becomes a fairness issue. A tokenizer trained predominantly on English text develops merges optimized for English subword patterns. Non-English text gets worse compression:

| Language | Typical fertility (GPT-2 tokenizer) | Typical fertility (multilingual tokenizer) |
|----------|--------------------------------------|---------------------------------------------|
| English | ~1.2 | ~1.1 |
| Spanish | ~1.5 | ~1.2 |
| Chinese | ~2.5 | ~1.5 |
| Hindi | ~3.5 | ~1.8 |
| Thai | ~4.0+ | ~2.0 |

A 3x fertility gap means that for the same semantic content, a Thai user consumes 3x the context window, pays 3x the API cost (if billed per token), and receives 3x less content per inference call. The Qwen 3 report addresses this directly — its expansion to 151,669 tokens and 119 languages was driven by the need to reduce multilingual fertility gaps. LLaMA 3's jump to 128K tokens was similarly motivated by multilingual encoding efficiency.

---

## 4. Tokenization Failures

Tokenization is a brittle preprocessing step, and its failures propagate silently into model behavior. Three categories of failure deserve architectural attention.

### 4.1 Digit Splitting and Arithmetic

Most BPE tokenizers split multi-digit numbers inconsistently. "123456" might tokenize as "123" + "456" or "12" + "3456" or "1" + "234" + "56" depending on what merges the training corpus produced. This means the model has no consistent positional representation of digit places, making arithmetic unreliable.

Some models address this by forcing single-digit tokenization (each digit is its own token) or by adding special number-handling logic. LLaMA's tokenizer, for example, splits all digits into individual tokens. This costs sequence length for numbers but gives the model consistent digit-level granularity.

### 4.2 The SolidGoldMagikarp Phenomenon

In 2023, researchers discovered that certain tokens in GPT-2/3's vocabulary triggered bizarre model behavior: hallucinations, refusal to repeat the token, and incoherent outputs. The tokens included "SolidGoldMagikarp" (a Reddit username), " TheNitromeFan", and "clostridium". These were dubbed "glitch tokens."

The mechanism: BPE constructs vocabulary from training corpus statistics. If a string appears frequently enough during BPE training to earn its own token but then appears rarely or never in the actual model training data, the model's embedding for that token receives minimal gradient signal. The embedding is essentially untrained — a random vector in a high-dimensional space. When the model encounters this token at inference time, the untrained embedding produces unpredictable activations, leading to undefined behavior.

**Why this matters architecturally:** The BPE vocabulary is learned from a tokenizer training corpus that may differ from the model training corpus. GPT-2's BPE was trained on WebText, and certain usernames or niche terms occurred frequently enough to become tokens. But during model training, those tokens might have been filtered out or appeared in different contexts. The result is a vocabulary with "dead" entries — tokens that exist but have no learned semantics.

**Mitigation strategies:**
- Train the tokenizer on exactly the same corpus used for model training (or a representative subset).
- Post-hoc audit: identify tokens with low training frequency and either remove them or initialize their embeddings from similar tokens.
- Regularization: ensure all embeddings receive some gradient signal, even for rare tokens.

### 4.3 Whitespace and Formatting Sensitivity

Tokenizers are sensitive to whitespace in ways that surprise users. " Hello" (with a leading space) and "Hello" are different tokens. The number of spaces in code indentation produces different tokenizations. Python code indented with spaces vs. tabs tokenizes completely differently, which can cause inconsistent code generation quality.

This sensitivity arises because byte-level BPE treats the space byte (0x20) as just another byte — it has no special status. Merges involving space-prefixed tokens (like " the", " is", " of") are among the most common in English BPE vocabularies, since most words appear after a space. The model effectively learns that " the" (with space) and "the" (without) are different vocabulary items with different distributions.

---

## 5. Impact on Model Architecture

Tokenization decisions propagate through the architecture in ways that constrain or enable other design choices.

### 5.1 Positional Encoding Pressure

Shorter sequences from better tokenization reduce the demands on positional encoding schemes. A model with a 128K vocabulary and fertility rate of 1.0 tokens/word can represent a 4,000-word document in ~4,000 tokens. The same document with a 32K vocabulary and fertility rate of 1.3 requires ~5,200 tokens. The first model needs positional encodings that generalize to 4K positions; the second needs them to generalize to 5.2K. At the scale of 128K context windows ([[ch-16]]), this compression directly determines how much actual content fits within the model's attention range.

### 5.2 Output Softmax Bottleneck

At every generation step, the model computes a softmax over the full vocabulary to produce the next-token distribution. For a 151K vocabulary (Qwen 3), this means a matrix multiply of $[d_\text{model}] \times [151,669]$ followed by a softmax over 151K values. The compute cost is linear in vocabulary size, and it occurs at every decoding step.

For large models, this cost is dominated by the Transformer layers. But for efficient inference with small models or speculative decoding, the output projection becomes a non-trivial fraction of per-token latency. This is another reason small models (like Qwen3-0.6B) tie embeddings — it halves the weight memory for the vocabulary projection.

### 5.3 Embedding Dimension and Vocabulary Coupling

The embedding dimension must be expressive enough to distinguish all vocabulary entries. With a 32K vocabulary, a 4096-dimensional embedding space has ~128 dimensions per token (in the crude ratio $d_\text{model} / V$). With 128K vocabulary, that ratio drops to ~32. In practice, embeddings are not partitioned this way — they share a continuous space — but the intuition holds: a larger vocabulary requires the embedding space to carve out more distinct regions, potentially pressuring the model to use wider layers or deeper representations.

---

## Core Insights from the Literature

### Insight 1: Vocabulary size is a scaling-law variable, not a hyperparameter
**Source:** LLaMA 3 Technical Report (Meta, 2024)

LLaMA 3's jump from 32K to 128K vocabulary was driven by scaling experiments that showed the 4x expansion improved encoding efficiency enough to offset the embedding parameter cost — at 405B scale, the 128K embedding table is 0.26% of total parameters but reduces token counts by 15-20% for English and 30-50% for non-English languages. This means vocabulary size should be chosen as part of the scaling law optimization, not fixed at a conventional value. **Guideline:** When designing a model family across scales, evaluate vocabulary size jointly with model width and depth. The optimal vocabulary is larger than most practitioners assume, especially for multilingual models.

### Insight 2: Tokenizer-model mismatch creates invisible failure modes
**Source:** SolidGoldMagikarp investigation (Rumbelow & watkins, 2023); GPT-2 paper (Radford et al., 2019)

GPT-2's byte-level BPE was a major advance (no UNK tokens, open-vocabulary coverage), but the BPE merge table was learned on a different data distribution than the model's training data. Tokens earned through BPE frequency that then appeared rarely during model training became "glitch tokens" — embeddings with near-random values that cause undefined behavior. The failure mode is invisible: the model does not signal that it has encountered an untrained token. It simply produces garbage. **Guideline:** The tokenizer training corpus must be representative of the model training corpus. Audit the vocabulary for tokens with low training-time frequency and treat them as a source of silent failures.

### Insight 3: Multilingual fairness is a tokenization problem before it is a model problem
**Source:** Qwen 3 Technical Report (Alibaba, 2025); LLaMA 3 Technical Report (Meta, 2024)

Both LLaMA 3 and Qwen 3 expanded their vocabularies specifically to improve non-English encoding efficiency. The fertility gap (English ~1.2 tokens/word vs. Hindi ~3.5 tokens/word on an English-centric tokenizer) means non-English users receive fundamentally worse service: less context per sequence, more compute per semantic unit, higher API costs. This is not a model capability issue — the model may understand Hindi perfectly — but a tokenization efficiency issue that penalizes non-English text at the infrastructure level. Qwen 3's expansion to 119 languages with 151K vocabulary is a direct response. **Guideline:** Measure fertility rate per target language as a first-order evaluation metric. If any language's fertility is >2x the English baseline, the tokenizer needs rebalancing.

### Insight 4: Deterministic tokenization is an underappreciated constraint
**Source:** Kudo, "Subword Regularization" (2018)

BPE produces a single deterministic segmentation for any input string. The unigram model (used in SentencePiece) can produce multiple segmentations with calibrated probabilities, enabling subword regularization — sampling different tokenizations of the same text during training. Kudo showed this improves robustness, especially on low-resource languages where the training corpus is too small for BPE to learn stable merges. The architectural implication: models trained with BPE have a blind spot for alternative segmentations, while unigram-based models develop representations that are more robust to tokenization variation. **Guideline:** For multilingual or low-resource settings, prefer SentencePiece unigram over BPE, and enable subword regularization during training.

---

## Key Takeaways

1. **BPE builds vocabulary bottom-up by iteratively merging the most frequent adjacent pairs.** It is greedy, frequency-driven, and linguistically indifferent. Byte-level BPE (GPT-2+) eliminates UNK tokens by operating on raw bytes.

2. **WordPiece uses likelihood-based merging (PMI) rather than raw frequency,** while the SentencePiece unigram model works top-down by pruning a large candidate vocabulary. SentencePiece handles CJK languages natively without pre-tokenization.

3. **Vocabulary size is a first-order architectural decision.** It determines embedding table cost ($V \times d_\text{model}$ parameters), sequence length (larger vocab = shorter sequences = cheaper attention), and multilingual fairness (higher vocab = more equitable fertility rates).

4. **The SolidGoldMagikarp phenomenon shows that tokenizer-model corpus mismatch creates invisible failure modes.** Tokens with learned BPE entries but rare model training occurrences become untrained embeddings that produce undefined behavior.

5. **Fertility rate (tokens per word) varies 2-4x across languages on English-centric tokenizers.** This is a fairness issue: non-English users pay more compute, consume more context, and receive worse compression.

6. **For small models (sub-8B), vocabulary cost dominates.** LLaMA 3 8B's untied 128K embeddings consume 13% of total parameters. Weight tying and vocabulary sizing must be co-optimized with model scale.

7. **Tokenization decisions constrain positional encoding, output softmax cost, and effective context length.** Better compression means more content per context window, which directly impacts long-context performance ([[ch-16]]).

---

## References

- Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units" (BPE for NMT, 2016)
- Gage, "A New Algorithm for Data Compression" (Original BPE, 1994)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, byte-level BPE, 2019) — [[gpt-2|paper]]
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (WordPiece, 2018) — [[bert|paper]]
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer" (2018)
- Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" (2018)
- Meta AI, "The Llama 3 Herd of Models" (128K vocabulary, 2024) — [[llama-3|report]]
- Qwen Team, "Qwen3 Technical Report" (151K vocabulary, 119 languages, 2025) — [[qwen-3|report]]
- Rumbelow & watkins, "SolidGoldMagikarp: Understanding Glitch Tokens" (2023)
- Schuster & Nakajima, "Japanese and Korean Voice Search" (WordPiece, 2012)
