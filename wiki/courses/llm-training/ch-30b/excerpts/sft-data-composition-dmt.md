---
chapter: ch-30b
course: llm-training
phase: read
excerpt_of: primary source (no library card existed on 2026-09-15)
source_url: https://arxiv.org/abs/2310.05492
created_at: "2026-09-15"
---

# Excerpt: How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition

- **Authors:** Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, et al. (Alibaba Group)
- **Year:** arXiv v1 2023-10; v4 2024-06; ACL 2024
- **Checked against:** arXiv:2310.05492v4 (PDF, full text including Appendices A-H), 2026-09-15
- **Why this excerpt exists:** ch-30b cites this paper for interference between math, code, and general-chat SFT data. The library card `sft-data-composition-dmt` was planned but not present when the chapter was written, so this file records the quoted numbers with their loci.

## Setup (§3.1; App. A, C)
- Base models: LLaMA 7B, 13B, 33B. Framework: FastChat.
- Data (App. A, Table 2), full size at k = 1: GSM8K RFT 110,142 examples (7.5K questions); Code Alpaca 20,022; ShareGPT 86,060.
- Subsets used for scaling: k ∈ {1, 1/4, 1/16, 1/64, 1/256}. At k = 1/256 the counts are 430, 78, and 336 (Table 2).
- Evaluation: GSM8K test set (greedy, maj@1), HumanEval (pass@1, greedy), MT-Bench (GPT-4 judge; three runs averaged, App. C).
- Training: 3 epochs, peak LR 2e-5, 3% warmup, final-epoch checkpoint (§3.1; App. C).
- Batch size conflict inside the paper: "The batch size during SFT is 16" (§3.1) versus "a batch size of 128 on NVIDIA A100 GPUs" (App. C). The paper does not reconcile the two values.

## Findings with loci
- **RQ1 (§3.2, Fig. 2).** Math and code scores keep rising with data amount; the general (MT-Bench) score "emerges with only around 1k data samples (ranging from 1/256 to 1/64)" and then rises slowly.
- **RQ2 (§3.3, Fig. 3).** Mixing the three sources improves each ability at low resource (1/256) and lowers it at high resource (k = 1) relative to training on that ability's data alone. For 7B the crossover lies between 1/64 and 1/16.
- **RQ3 (Fig. 4).** With ShareGPT fixed and code+math data scaled (or the reverse), the math score shows "minimal impact" from the ratio; the code score fluctuates, which the authors attribute to code data inside ShareGPT (Interpretation). Summary statement: "data amount directly influences performance conflicts, whereas the impact of data ratio is insignificant within our experimental setup" (§5 Conclusion).
- **Ablation of code and math inside ShareGPT (§4.2, Fig. 6).** InsTag tags containing "code" or "math" are removed by regular expression, reducing ShareGPT from 86K to 63K examples. The mixed-source conflict at high resource is reduced "to some extent".
- **RQ4 and DMT (Table 1).** Four strategies: multi-task learning (one mixed stage), sequential training (code → math → general), mixed sequential training (code+math, then general), and DMT (code+math, then general plus a fraction k of code+math).

Table 1 (GSM8K / HumanEval / MT-Bench):

| Method | LLaMA-7B | LLaMA-13B | LLaMA-33B |
|---|---|---|---|
| General only | 11.10 / 10.42 / 5.88 | 14.02 / 16.40 / 6.13 | 26.06 / 24.30 / 6.63 |
| Math only | 49.10 / 6.71 / 2.53 | 51.40 / 12.8 / 2.54 | 57.91 / 15.5 / 3.18 |
| Code only | 4.51 / 18.40 / 4.30 | 5.15 / 17.1 / 3.53 | 6.06 / 26.82 / 4.18 |
| Multi-task learning | 47.53 / 14.63 / 5.76 | 50.94 / 19.50 / 5.73 | 56.69 / 18.9 / 6.07 |
| Sequential training | 31.39 / 15.85 / 5.72 | 39.12 / 20.12 / 5.93 | 47.27 / 24.80 / 6.73 |
| Mixed sequential training | 32.60 / 15.24 / 6.02 | 40.48 / 18.30 / 5.93 | 44.24 / 24.4 / 6.43 |
| DMT (k = 1/256) | 41.92 / 17.68 / 6.08 | 46.47 / 19.50 / 6.03 | 56.36 / 25.00 / 6.73 |

- The prose gives the 33B HumanEval change as "24.4 to 25.5"; Table 1 prints 25.00. The prose gives the 33B MT-Bench change as "6.43 to 6.69"; Table 1 prints 6.73 (§3.5).
- **k sweep (§4.3, Fig. 5).** Raising k from 0 to 1/256 improves both specialized and general scores; from 1/256 to 1/4 the general and specialized scores move in opposite directions; from 1/4 to 1 the general score declines. The authors state that k "needs to be tuned based on specific requirements".
- **Equal amount vs equal proportion (App. H, Table 6).** Mixing equal example counts versus equal subset proportions gives similar results at k = 1/16, 1/64, 1/256 (7B).
- **OOD check (App. F, Table 5).** MATH and MBPP follow the same low-resource-gain and high-resource-conflict pattern at 7B.

## Limits stated or visible in the source
- LLaMA-1 base models only; three abilities, one benchmark each; MT-Bench scored by GPT-4 (Limitations).
- Seeds or confidence intervals are not reported except the three-run average for MT-Bench (App. C).
- The t-SNE analysis (§4.1, Fig. 5) is qualitative: code and general representations overlap, math representations separate.
