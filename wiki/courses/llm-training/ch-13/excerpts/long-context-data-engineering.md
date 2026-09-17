---
chapter: ch-13
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/long-context-data-engineering.md (library card has no Verification section and contains values not found in the paper; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2402.10171
primary_version: arXiv:2402.10171v1 (2024-02-15)
created_at: "2026-09-15"
---

# Excerpt: Data Engineering for Scaling Language Models to 128K Context

Authors: Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, Hao Peng (University of Edinburgh, MIT-IBM Watson AI Lab, University of Melbourne, Ohio State University, University of Washington, MIT, UIUC). Source type: paper. Read in the v1 PDF text on 2026-09-15 for ch-13 `read.md` §6, §7, and Recipe.

## Main claim (Abstract)
> "we find that naïvely upsampling longer data on certain domains like books, a common practice of existing work, gives suboptimal performance, and that a balanced domain mixture is important. We demonstrate that continual pretraining of the full model on 1B-5B tokens of such data is an effective and affordable strategy for scaling the context length of language models to 128K."

## Data strategies (§3, Figure 2)
Source data: SlimPajama, "consisting of 82% web data (67% from CommonCrawl and 15% from C4), 4.5% code (Github), 4.5% Wikipedia, 4.5% books, 2.5% Arxiv, and 2.0% StackExchange."

> "The documents' lengths and their source domains are two closely related confounding factors in data engineering because long data usually come from particular sources."

- Cut at 4K: "Since there are about 30% documents that are naturally longer than 4K, this approach breaks such naturally-existing long-range dependencies."
- Cut at 128K: preserves naturally existing long dependencies without changing the domain mixture.
- Per-source Upsampling: "This retains the domain mixture, then upsamples long documents within each domain."
- Global Upsampling: "This upsamples long documents while ignoring their source domains, and consequently slightly changes the domain mixture."
- Upsample Arxiv/Book/Github: "This approach changes both the domain and length distributions."

§5.3: "Then in each of the domains, we upsample sequences longer than 4K from about 30% to about 70%. Doing this enables us to keep the domain mixture ratio fixed, only changing the length distribution of the training documents."

## Training settings (§4, Table 2)
LLaMA-2 7B at 80K and 13B at 64K sequence length; "constant learning rate 2e-5"; RoPE base modified "as in Xiong et al. (2023)" (value not printed); data packed "to 80K chunks regardless of the document boundary"; "batch size to be 4M tokens"; "5B tokens, which translates to ... 2000 optimization steps". Hardware: 8 × 80G A100.

## Results
- Table 3 (Needle / MMLU): Ours LLaMA-2 7B (80K) 88.0 / 43.3; LongLoRA 7B (100K) 70.0 / 37.9; Together LLaMA-2 7B 32K 27.9 / 44.8; YaRN Mistral 7B 128K 57.4 / 59.4; Ours LLaMA-2 13B (64K) 90.0 / 52.4; GPT-4-Turbo 128K 87.1 / 86.4. The base LLaMA-2 MMLU is not listed in Table 3.
- Data quantity (Figure 3): 500M tokens "enough to unlock most of the retrieval accuracy"; retrieval saturates at about 5B tokens; 10B tokens gave 84.0 vs 88.0 at 5B.
- Figure 4: the original mixture "despite giving very close loss, performs badly on precise retrieval. Per-source length upsampling significantly improves precise retrieval."

## Table 5 (loss difference against the original mixture; 7B, 5B tokens packed to 80K; > 0.01 treated as significant)
| 0-4K context | C4 | CC | Stack | Arxiv | Wiki | Book | Github |
|---|---|---|---|---|---|---|---|
| Original (loss) | 2.038 | 1.760 | 1.519 | 1.660 | 1.424 | 2.085 | 0.907 |
| Per-source | +.002 | +.008 | −.001 | −.008 | −.040 | −.065 | −.008 |
| Global | +.008 | +.010 | +.015 | −.020 | −.020 | −.140 | +.015 |
| Code↑ | +.010 | +.016 | +.010 | +.006 | −.026 | +.030 | −.023 |
| Book↑ | +.010 | +.016 | +.021 | +.000 | −.010 | −.175 | +.029 |
| Arxiv↑ | +.006 | +.016 | +.013 | −.060 | −.030 | +.040 | +.025 |

| 4K-128K context | C4 | CC | Stack | Arxiv | Wiki | Book | Github |
|---|---|---|---|---|---|---|---|
| Original (loss) | 1.560 | 1.650 | 0.786 | 1.075 | 1.313 | 1.852 | 0.447 |
| Per-source | −.010 | −.010 | −.006 | −.011 | −.044 | −.014 | +.002 |
| Global | −.010 | −.006 | −.001 | −.016 | −.040 | −.018 | −.007 |
| Code↑ | −.008 | −.002 | −.003 | −.007 | −.042 | −.010 | −.029 |
| Book↑ | −.010 | −.006 | +.001 | −.007 | −.037 | −0.30 (as printed) | +.000 |
| Arxiv↑ | −.008 | −.002 | +.002 | −.036 | −.039 | −.010 | −.004 |

> "upsampling one domain, e.g., code, may even harm another domain, e.g., book. Per-source length upsampling is the most balanced mixture with almost no significant increase of loss across domains." (Table 5 caption)

## Corrections to the library card (found while building this excerpt; the card was not edited)
- "documents longer than 32K get 5× weight" → the paper raises the share of sequences longer than 4K from about 30% to about 70% within each source (§5.3); no 5× factor or 32K threshold is stated.
- "RoPE base-θ rescaled from 10K to 200M"; "LR 2e-5 → 2e-6 cosine" → RoPE base value not printed; constant LR 2e-5 (§4).
- "context window 80K ... NIAH ... >99% at 128K"; "MMLU within 1 point of base"; "MMLU drops 3–5 points if cross-domain proportions are changed"; "~30K A100-hours" → not found; Table 3 gives Needle 88.0 and MMLU 43.3 for the 7B model, and Table 5 gives per-domain loss changes, not MMLU changes.

## Later comparison (from [[prolong]], App. B.6, Table 24; §2.3, Table 2)
In ProLong's matched setting on Llama-3-8B, this SlimPajama long mix scored 51.8 long / 65.4 short against 54.6 / 67.5 for ProLong's mix, and fine-tuning on it lowered MMLU from 66.5 to 63.1 and GSM8K from 44.7 to 40.6.
