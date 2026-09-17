---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: SmolLM2 paper (SmolTalk composition and ablations), SmolTalk dataset card, and Tülu 3 SFT data ablations (library cards [[smol-talk]] and [[tulu-3]] are not verified; quotations below are taken from the primary sources)
source_url: https://arxiv.org/abs/2502.02737 ; https://huggingface.co/datasets/HuggingFaceTB/smoltalk ; https://arxiv.org/abs/2411.15124
primary_version: SmolLM2 arXiv:2502.02737 (2025-02); Tülu 3 arXiv:2411.15124v5 (v1 2024-11; v5 2025-04)
created_at: "2026-04-23"
revised_at: "2026-09-15 (rewritten from the primary sources for the 2026-09 revision of read.md)"
---

# Excerpt: Multi-turn data in released SFT mixes — SmolTalk and Tülu 3

Used by ch-25 `read.md` §3, §7, §9, the Recipe, and the Generalization lens. The earlier version of this excerpt quoted the unverified [[system-prompt-diversity]] card (2× vocabulary, 3× follow-ups, +5–10 IFEval); those numbers have no primary source and are removed.

## SmolTalk composition (SmolLM2 App. F, Table 9: "The total dataset contains 1.1M instruction-response pairs")
| Group | Dataset source | Samples |
|---|---|---|
| New | MagPie-Ultra | 431k |
| New | Smol-Rewrite | 56.2k |
| New | Smol-Constraints | 36.2k |
| New | Smol-Summarization | 101k |
| Math | NuminaMath-CoT | 112k |
| Math | MetaMathQA | 50k |
| Other | Self-OSS-Starcoder2-Instruct | 50.7k |
| Other | APIGen-Function-Calling | 87.5k |
| Other | SystemChats2.0 | 35.9k |
| Other | LongAlign | 3.73k |
| Other | Everyday-Conversations | 2.38k |
| Other | Explore-Instruct-Rewriting | 32k |
| Other | OpenHermes2.5 | 100k |

The components sum to 1,098.61k (derived). The dataset card gives rounded or different sizes (Smol-Magpie-Ultra 400K, Smol-rewrite 50k, Smol-summarize 100k, SystemChats 30k, APIGen 80k); the paper table describes the trained mix.

## MagPie-Ultra is multi-turn (SmolLM2 §5.1.1)
> "MagPie-Ultra is a multi-turn dataset created using the two-step prompting method from (Xu et al., 2024). Unlike MagPie, which used Llama-3-70B-Instruct without specific system prompts to generate two-turn conversations, MagPie-Ultra leverages the larger, more powerful model Llama-3.1-405B-Instruct-FP8 (Dubey et al., 2024). We also incorporate system prompts to guide generation, producing a balanced dataset of 1M samples with three-turn conversations. The resulting dataset was further filtered using smaller Llama models (Llama-3.1-8B-Instruct and Llama-Guard-3-8B) to ensure quality and safety of the generated instructions. We also leveraged ArmoRM (Wang et al., 2024b;a) to score conversations for quality-based filtering, and gte-large-en-v1.5 (Zhang et al., 2024; Li et al., 2023c) to deduplicate semantically similar conversations."

Everyday-Conversations is described as "2.2k casual multi-turn interactions" (§5.1.4). Derived lower bound on the multi-turn share: (431 + 2.38) / 1,098.61 = 39.4%, not counting components whose turn structure the paper does not state.

## SmolTalk dataset card statements
- SystemChats2.0: "to make the model support a variety of system prompt formats we add 30k samples from the SystemChat-2.0 dataset. Note that Smol-rewrite and and Smol-summarize datasets also include system prompts."
- LongAlign: "we find that finetuning the model on only short samples makes it loose long context abilities beyond 2048 tokens, so we add english samples (with less than 16k tokens) from the LongAlign-10k dataset and train with a 8192 sequence."

## SmolLM2 Table 10 (SmolLM2 base fine-tuned 1 epoch per dataset; SmolLM2-SFT† 2 epochs on SmolTalk)
| Dataset | IFEval | MT-Bench | GSM8K | MATH | ARC-C | MMLU-Pro |
|---|---|---|---|---|---|---|
| OpenHermes | 30.01 | 1.02 | 42.91 | 12.76 | 40.27 | 20.32 |
| UltraChat | 27.26 | 4.66 | 30.40 | 9.06 | 41.21 | 15.79 |
| MagPie-Pro (single-turn) | 30.45 | 4.31 | 14.56 | 6.64 | 36.01 | 12.19 |
| MagPie-Pro-MT (multi-turn) | 31.66 | 5.40 | 20.55 | 7.84 | 36.69 | 11.97 |
| MagPie-Ultra | 35.49 | 5.22 | 24.34 | 13.56 | 37.71 | 12.01 |
| MagPie-Ultra+ (with Smol-Constraints, -Rewrite, -Summarization) | 48.16 | 5.28 | 19.94 | 12.74 | 38.91 | 12.43 |
| SmolTalk | 46.67 | 5.49 | 43.75 | 18.60 | 40.02 | 18.19 |
| SmolLM2-SFT† | 57.09 | 6.11 | 47.54 | 19.64 | 42.49 | 19.06 |

## Tülu 3 SFT data (arXiv:2411.15124v5)
- Table 7 lists WildChat (GPT-4 subset) with 241,307 original prompts and 100,000 used in SFT; the final SFT mix has 939,344 prompts.
- §4.2: "In our mix we also emphasized adding diverse chat data, mainly from WildChat. We show the impact of removing WildChat in Table 10, and we see that there is a small but noticeable degradation on most skills, most noticeably on Alpaca Eval, highlighting the importance of diverse real-world data."
- §4.2: "We also found that adding contrastive prompts, such as those in CoCoNot, were helpful for preventing our models from over-refusing safe prompts."

Table 10 (development evaluations; 8B):
| Model | Avg. | MMLU | TQA | PopQA | BBH | CHE | CHE+ | GSM | DROP | MATH | IFEval | AE 2 | Safety |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tülu 3 8B SFT | 60.1 | 62.1 | 46.8 | 29.3 | 67.9 | 86.2 | 81.4 | 76.2 | 61.3 | 31.5 | 72.8 | 12.4 | 93.1 |
| → w/o WildChat | 58.9 | 61.0 | 45.2 | 28.9 | 65.6 | 85.3 | 80.7 | 75.8 | 59.3 | 31.8 | 70.1 | 7.5 | 95.2 |
| → w/o Safety | 58.0 | 62.0 | 45.5 | 29.5 | 68.3 | 84.5 | 79.6 | 76.9 | 59.4 | 32.6 | 71.0 | 12.4 | 74.7 |
| → w/o Persona Data | 58.6 | 62.4 | 48.9 | 29.4 | 68.3 | 84.5 | 79.0 | 76.8 | 62.2 | 30.1 | 53.6 | 13.5 | 93.9 |
| → w/o Math Data | 58.2 | 62.2 | 47.1 | 29.5 | 68.9 | 86.0 | 80.5 | 64.1 | 60.9 | 23.5 | 70.6 | 12.0 | 93.5 |

Table 32 (development vs unseen; Inst. Follow. columns IFE = IFEval, IFEO = IFEval-OOD):
| Model | Dev. Avg | Unseen Avg. | IFE | IFEO |
|---|---|---|---|---|
| Tülu 3 8B SFT | 64.1 | 29.9 | 72.8 | 17.6 |
| w/o WildChat | 62.8 | 28.8 | 70.1 | 20.8 |
| w/o Safety | 63.7 | 29.7 | 71.0 | 17.6 |
| w/o Persona Data | 59.8 | 29.4 | 53.6 | 18.0 |
| w/o Math Data | 62.2 | 27.4 | 70.6 | 18.3 |

> "We see that the data choices generalize on average, as indicated by the best average performances on both development and unseen evaluations by the final SFT checkpoint. In individual skills, we see that our choices overfit to the development evaluations in Precise Instruction Following, and to some extent in Knowledge Recall and Reasoning." (App., text accompanying Table 32)

SFT hyperparameters (Table 11): 8B learning rate 5e-6, linear schedule, warmup ratio 0.03, effective batch 128, max token length 4,096, 2 epochs.
