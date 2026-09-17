---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: arXiv:2402.17463v2 (no library card at the time of writing; chapter-local verified extract)
source_url: https://arxiv.org/abs/2402.17463
created_at: "2026-09-15"
---

# Excerpt: Training-Free Long-Context Scaling of Large Language Models (Dual Chunk Attention)

- **Authors:** Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, Lingpeng Kong
- **Year:** 2024 (arXiv v1 2024-02; v2 2024-05-29)
- **Source type:** paper (code: github.com/HKUNLP/ChunkLlama)
- **Used in:** ch-32b §2.7, §3

## What DCA changes (§3)
DCA does not change weights. At inference it replaces the position indices used inside RoPE so that no
query-key relative distance exceeds the pretraining length c. The sequence is split into chunks of size s.
- **Intra-chunk attention** (Eq. 2): keys and queries in the same chunk use positions 0 … s−1.
- **Inter-chunk attention** (Eq. 5): for keys in chunks more than one chunk back, every query gets the same
  position c − 1, so the relative distance is c − 1 − P_k[j] ≥ c − s (Eq. 6).
- **Successive-chunk attention** (Eq. 7): for the immediately preceding chunk, the first w query positions are
  set to s, s+1, …, s+w−1 and the rest to c−1, which keeps the w nearest keys at the closest distances.
  "w means the local window size and can be directly set to the difference between pretraining length and
  chunk size c−s."
- Worked example printed in Fig. 2: c = 10, s = 6, w = 4; P_q^Succ = [6, 7, 8, 9, 9, 9, 6, 7, 8, 9, 9, 9].
- "The chunk size s can be typically set to 3/4 training length and for Llama2, this value is 3072" (§4).
- DCA is compatible with FlashAttention 2 (abstract; §4).

## Results quoted in ch-32b
- PG19 validation perplexity (Table 1, v2): Llama2 7B without any change: 7.87 at 4096 and >10² at 8192.
  ChunkLlama2 70B: 5.24 / 5.18 / 5.21 / 5.30 / 5.59 at 4096 / 8192 / 16384 / 32768 / 65536.
  Llama2-PI 7B (training-free, dynamic factor): 9.19 at 8192 and 15.11 at 16384.
- Llama2 70B with DCA extends to 96K tokens "with only a minor increase of 0.56 PPL compared to its original
  performance at a 4k context length" (§4.1, Table 2).
- CodeLlama and Together's Llama2 fork scale to a 192K context with chunk size 24K (§4.1).
- The Llama2 70B chat model with DCA at a 16K window reaches "94% of the performance of gpt-3.5-turbo-16k"
  on the long-context research benchmarks of Table 3 (§4.2).

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2402.17463 (v2 PDF): abstract, §3 Eq. 2–7, Fig. 2 caption,
  §4 Table 1, §4.1, §4.2.
- Not reported by the source: short-context benchmark scores with DCA enabled. By Eq. 2, an input no longer than
  one chunk s gets its original positions (derived from the definition, not measured in the paper).
