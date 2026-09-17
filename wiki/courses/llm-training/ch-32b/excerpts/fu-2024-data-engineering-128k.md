---
chapter: ch-32b
course: llm-training
phase: read
excerpt_of: arXiv:2402.10171v1 (chapter-local verified extract; used instead of the library card long-context-data-engineering, which had not been revised against the primary source when this chapter was written)
source_url: https://arxiv.org/abs/2402.10171
created_at: "2026-09-15"
---

# Excerpt: Data Engineering for Scaling Language Models to 128K Context

- **Authors:** Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng
- **Year:** 2024 (arXiv v1 2024-02)
- **Source type:** paper (code: github.com/FranxYao/Long-Context-Data-Engineering)
- **Used in:** ch-32b §4, §5, Recipe

## Claim (abstract)
Long-context modeling "is a capability that is mostly already acquired through large-scale pretraining" and can be
extended "through lightweight continual pretraining on appropriate data mixture". "500 million to 5 billion tokens
are enough to enable the model to retrieve information anywhere within the 128K context."

## Data mixture (§3, §5.3)
- Per-source upsampling "keeps the mixture ratio of the data sources the same as the original data, i.e., 67%
  CommonCrawl (CC), 15% C4, 4.5% Github, 4.5% Wikipedia, 4.5% books, 2.5% Arxiv and 2.0% StackExchange for
  SlimPajama. Then in each of the domains, we upsample sequences longer than 4K from about 30% to about 70%."
- Alternatives compared: global upsampling of long sequences, and upsampling Arxiv, Book, or Github.
- Table 5 (7B, 5B tokens at 80K; validation loss minus loss of the original mixture; |Δ| > 0.01 treated as
  significant). At 0–4K context: per-source +.002 (C4), +.008 (CC), −.001 (Stack), −.008 (Arxiv), −.040 (Wiki),
  −.065 (Book), −.008 (Github); Book↑ +.010, +.016, +.021, +.000, −.010, −.175, +.029; Code↑ +.010, +.016, +.010,
  +.006, −.026, +.030, −.023. "upsampling one domain, e.g., code, may even harm another domain, e.g., book.
  Per-source length upsampling is the most balanced mixture with almost no significant increase of loss across
  domains."

## Training configuration (§4)
"For training, we use a constant learning rate 2e-5. We modify the base of RoPE positional encoding to adjust it
to longer context, as in Xiong et al. (2023). We pack all data to 80K chunks regardless of the document boundary
... We set the batch size to be 4M tokens ... We train the model on 5B tokens, which translates to 5B (size of
data) / 4M (batch size) = 2000 optimization steps." Sequence length is 80K for 7B and 64K for 13B. The RoPE base
value is not printed in the paper. Table 2: 7B at 80K on 8×80G A100 takes 10 days per 10B tokens ("5 days" for
the 5B-token run, §5.2).

## Results
- Table 3 (Needle / MMLU): Ours LLaMA-2 7B 80K 88.0 / 43.3; Ours LLaMA-2 13B 64K 90.0 / 52.4; LongLoRA 7B 100K
  70.0 / 37.9; YaRN Mistral 7B 128K 57.4 / 59.4; GPT-4-Turbo 128K 87.1 / 86.4. The base LLaMA-2 7B and 13B MMLU
  values are not printed in this table.
- Data quantity (Fig. 3, 7B NIAH accuracy): 100M 37.8; 300M 59.0; 500M 81.1; 1B 85.3; 5B 88.0; 10B 84.0. "At 10B
  token, the model seems to overfit on its 80K training range, and the length generalization starts to decrease."
- Table 4 (InfiniBench BookQA at 128K): 7B 27.4; 13B 31.1; GPT-4-Turbo 37.4.

## Verification
- Checked on 2026-09-15 against https://arxiv.org/abs/2402.10171 (v1 PDF): abstract, §3, §4, §5.1–5.3, Tables 2–5, Fig. 3.
- Corrections relative to the library card long-context-data-engineering (not edited by this chapter): the paper
  uses a constant LR 2e-5 (not cosine), upsamples documents longer than 4K (not "5× above 32K"), prints no RoPE base
  (not "200M"), reports NIAH 88.0 for 7B (not "~98%"), and reports no "3–5 point MMLU drop" ablation.
