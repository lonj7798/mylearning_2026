---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: Harm de Vries, "In the long (context) run" (no library card at the time of writing; chapter-local verified extract)
source_url: https://www.harmdevries.com/post/context-length/
created_at: "2026-09-15"
---

# Excerpt: In the long (context) run

- **Author:** Harm de Vries
- **Date:** last updated 2023-09-16
- **Source type:** practitioner evidence (FLOPs derivation plus length histograms of 10K sampled documents per source)
- **Used in:** ch-32b §1.4

## Attention cost (§1 and Appendix)
FLOPs per token for N_l layers, hidden size d, context length L:
- FLOP_FFN = N_l(48d²); FLOP_QKVO = N_l(24d²); FLOP_Att = N_l(6d(L+1)).
- For LLaMA-7B (d = 4096, N_l = 32) the attention share relative to FFN + QKVO is 8% at 4K and 260% at 128K.
  "employing an 8-16K context window leads to a manageable 16-33% overhead."
- "Both the FFN FLOPs and QKVO FLOPs ... grow quadratically with the hidden state dimension d", so for LLaMA-65B
  (d = 8192) a 16–32K window has the same 16–33% overhead.

## Length of pre-training documents (§2)
- C4 and RefinedWeb: "over ~95% of them containing fewer than 2K tokens."
- "Almost 45% of the tokens in RefinedWeb are derived from files exceeding 2K tokens."
- "over 12.5% of the tokens in RefinedWeb are from files exceeding 16K tokens, while it is not even 2.5% for C4."
- starcoderdata: "over 80% have fewer than 3K tokens"; for C, "over 50% of the tokens originate from files
  exceeding 16K tokens" although they are "less than 5% of the files".
- Wikipedia: "more than 50% of Wikipedia articles consist of over 4,000 tokens"; Gutenberg books: "over 75% of
  these books contain more than 16,000 tokens".

## Author's conclusion (§3, Interpretation)
Packing random examples into a 16–32K window means "we would spend much of the compute overhead on tokens that do
not require communication between them", which "might also hurt performance as the model will be trained to
ignore other tokens within the sequence." Proposed directions: hyperlinks, repository structure, edit histories,
and variable sequence-length buckets. The post states that long-context benchmarks were lacking at the time (§3.4).

## Verification
- Checked on 2026-09-15 against https://www.harmdevries.com/post/context-length/ (page text as served on that date).
- Not reported by the source: controlled training experiments (the post measures data and FLOPs only).
