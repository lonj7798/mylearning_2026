---
chapter: ch-29e
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/urial.md (library card not present on 2026-09-15; quotations below are taken from the primary source)
source_url: https://arxiv.org/abs/2312.01552
primary_version: arXiv:2312.01552v1 (2023-12)
created_at: "2026-09-15"
---

# Excerpt: The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)

Verbatim quotations and table values used by ch-29e `read.md`, with loci. Checked against the v1 PDF on 2026-09-15.

## Token distribution shift (§2.1-2.2)
> "(1) unshifted positions (η = 1): o_t is the top-ranked token in both P_base and P_align [...] (2) marginal positions (1 < η ≤ 3) [...] (3) shifted positions (η > 3): in this case, o_t is rather unlikely to be sampled by P_base"

> "On average, across 1,000 examples that we tested, 77.7% of the tokens are at such unshifted positions, which increases to 92.2% when including marginal positions."

Figure 3 ratios (unshifted / marginal / shifted): Llama-2-7b → Llama-2-7b-chat 77.7% / 14.5% / 7.8%; Llama-2-7b → Vicuna-7b-v1.5 82.4% / 12.8% / 4.8%; Mistral-7b → Mistral-7b-instruct 82.2% / 12.5% / 5.2%.

> "We observe that shifted positions frequently consist of 'stylistic tokens', such as discourse markers and transitional words."

## Method (§3.3)
> "Together, the K=3 examples and the system prompt comprise a total of 1,011 tokens (or 671 words)."

## Evaluation (§4.1-4.2)
> "We use GPT-4 to evaluate the 800 regular instructions for evaluating the first five aspects, while ChatGPT is employed evaluate the 200 red-teaming and malicious instructions for the safety aspect."

> "We choose to use greedy decoding (i.e., zero temperature) in all experiments for reproducibility."

Table 1 averages (1-5 scale): Vicuna-7b (SFT) 4.46; Llama2-7b-chat (RLHF) 4.47; Llama2-7b URIAL K=3 4.33; Mistral-7b-instruct (SFT) 4.44; Mistral-7b URIAL K=3 4.63; Llama2-70b-chat^q (RLHF) 4.67; Llama2-70b^q URIAL K=3 4.74; gpt-3.5-turbo-0301 4.75; gpt-4-0613 4.80. (^q marks 4-bit GPTQ quantization.)

## Stated limits (§4.3, §5.4)
> "we find that Mistral-7B with URIAL can correctly answer the question 'Did Facebook corporate change its name?' by telling users the new name is 'Meta Platform Inc.'. However, the SFT-ed version Mistral-7B-Instruct instead answers 'No, Facebook did not change its name.'"

> "Although URIAL can match the performance of SFT and RLHF when the base LLMs are strong, it is not suggested to replace SFT or RLHF with URIAL in all scenarios. Specifically, model tuning may still be necessary for tasks such as coding (Luo et al., 2023), mathematics (Yue et al., 2023), interactive agents (Yin et al., 2023), etc."
