<!-- scope: BPE algorithm step-by-step with worked example, merge rules, encoding/decoding, parent: [[ch-05]] -->

# BPE Algorithm: Step-by-Step Walkthrough

Byte Pair Encoding is the tokenization algorithm behind GPT-2, GPT-3, GPT-4, LLaMA, and most frontier LLMs. This excerpt traces the full algorithm from corpus to vocabulary, with a worked example that makes each step concrete. Understanding BPE at this level is necessary for diagnosing tokenization failures and reasoning about vocabulary design decisions.

---

## Phase 1: Vocabulary Construction (Training Time)

BPE vocabulary construction is a greedy, iterative algorithm. It starts with the smallest possible vocabulary and grows it by merging the most frequent adjacent pairs.

### Input

A training corpus split into words (pre-tokenized by whitespace and punctuation). Each word is represented as a sequence of characters with a special end-of-word marker `</w>`.

### Worked Example

**Corpus with word frequencies:**
```
"hug"    : 10 occurrences
"pug"    :  5 occurrences  
"pun"    : 12 occurrences
"bun"    :  4 occurrences
"hugs"   :  5 occurrences
```

**Step 0 -- Initialize with character vocabulary:**
```
Vocab: {h, u, g, p, n, b, s, </w>}  (8 symbols)

Word representations:
  h u g </w>     (freq 10)
  p u g </w>     (freq  5)
  p u n </w>     (freq 12)
  b u n </w>     (freq  4)
  h u g s </w>   (freq  5)
```

**Step 1 -- Count all adjacent pairs:**
```
(h, u)    : 10 + 5 = 15
(u, g)    : 10 + 5 + 5 = 20
(g, </w>) : 10 + 5 = 15
(p, u)    : 5 + 12 = 17
(u, n)    : 12 + 4 = 16
(n, </w>) : 12 + 4 = 16
(b, u)    : 4
(g, s)    : 5
(s, </w>) : 5
```

**Winner: (u, g) with count 20. Merge into `ug`.**

```
Vocab: {h, u, g, p, n, b, s, </w>, ug}  (9 symbols)

  h ug </w>      (freq 10)
  p ug </w>      (freq  5)
  p u n </w>     (freq 12)
  b u n </w>     (freq  4)
  h ug s </w>    (freq  5)
```

**Step 2 -- Recount pairs:**
```
(h, ug)    : 10 + 5 = 15
(ug, </w>) : 10 + 5 = 15
(p, ug)    : 5
(p, u)     : 12
(u, n)     : 12 + 4 = 16
(n, </w>)  : 12 + 4 = 16
(b, u)     : 4
(ug, s)    : 5
(s, </w>)  : 5
```

**Winner: (u, n) with count 16. Merge into `un`.**

```
Vocab: {..., un}  (10 symbols)

  h ug </w>      (freq 10)
  p ug </w>      (freq  5)
  p un </w>      (freq 12)
  b un </w>      (freq  4)
  h ug s </w>    (freq  5)
```

**Step 3 -- (n, </w>) no longer exists. Recount:**
```
(h, ug)    : 15
(ug, </w>) : 15
(p, ug)    : 5
(p, un)    : 12
(un, </w>) : 12 + 4 = 16
(b, un)    : 4
(ug, s)    : 5
(s, </w>)  : 5
```

**Winner: (un, </w>) with count 16. Merge into `un</w>`.**

This continues until the target vocabulary size is reached (e.g., 50,257 for GPT-2).

---

## Phase 2: Tokenization (Inference Time)

Given the learned merge table (an ordered list of merge rules), tokenize new text by applying merges in priority order.

### Algorithm

1. Split the input into characters (or bytes, for byte-level BPE).
2. Scan the sequence for the highest-priority merge pair that exists.
3. Apply that merge everywhere it occurs.
4. Repeat until no more merges apply.

### Critical Property: Determinism

BPE tokenization is **deterministic** -- the same input always produces the same token sequence. The merge table has a fixed priority order (the order in which merges were learned during training). This means there is exactly one valid tokenization for any input string.

This determinism is a double-edged sword:
- **Advantage**: Reproducibility. The same text always maps to the same tokens.
- **Disadvantage**: No regularization. The model never sees alternative segmentations of the same word during training, which can make it brittle to tokenization-dependent patterns. This is why SentencePiece's unigram model (which can sample different segmentations) sometimes provides better robustness ([[ch-05]]).

---

## Byte-Level BPE (GPT-2+)

Standard BPE operates on Unicode characters, requiring the base vocabulary to include every possible character. Byte-level BPE ([[gpt-2|paper]]) operates on raw bytes instead:

- **Base vocabulary**: Exactly 256 entries (one per byte value, 0x00-0xFF)
- **Any text is representable**: Every Unicode character decomposes into 1-4 UTF-8 bytes
- **No UNK tokens**: Unlike character-level BPE, byte-level BPE never encounters an unknown symbol

GPT-2's final vocabulary: 256 byte tokens + 50,000 learned merges + 1 end-of-text token = **50,257 tokens**.

The tradeoff: rare Unicode characters (Chinese characters, emoji, mathematical symbols) may require multiple byte tokens, inflating sequence length for non-Latin scripts. A single Chinese character encoded in UTF-8 is 3 bytes; if none of those byte sequences have been merged into a single token, that character consumes 3 tokens instead of 1.

---

## Merge Table as Compressed Knowledge

The BPE merge table implicitly encodes the frequency structure of the training corpus. Common English patterns appear early:
- Merges 1-100: common character pairs ("th", "he", "in", "er", "an", ...)
- Merges 100-1000: common syllables and short words (" the", " of", " and", ...)
- Merges 1000-10000: complete common words and frequent subwords
- Merges 10000+: domain-specific terms, rare words, multi-word expressions

This ordering is the reason BPE's compression quality depends on training corpus representativeness. A merge table learned on English text will produce poor compression for Chinese, because byte sequences common in Chinese UTF-8 encoding will not have been merged.

---

## References

- Sennrich, Haddow & Birch, "Neural Machine Translation of Rare Words with Subword Units" (2016)
- [[gpt-2|Radford et al. "Language Models are Unsupervised Multitask Learners" (2019) (paper)]]
- Gage, "A New Algorithm for Data Compression" (1994)
