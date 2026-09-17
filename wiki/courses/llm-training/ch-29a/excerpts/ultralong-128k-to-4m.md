---
chapter: ch-29a
course: llm-training
phase: read
excerpt_of: arXiv:2504.06214v1 (From 128K to 4M: Efficient Training of Ultra-Long Context LLMs), §3, Tables 1-5 (chapter-local verified extract; no library card exists yet)
source_url: https://arxiv.org/abs/2504.06214
created_at: "2026-09-15"
---

# Excerpt: UltraLong-8B — separators without cross-document masking, short-only SFT

- **Authors:** Chejian Xu, Wei Ping, Peng Xu, Zihan Liu, Boxin Wang, Mohammad Shoeybi, Bo Li, Bryan Catanzaro (NVIDIA; UIUC)
- **Year:** 2025 (arXiv v1 2025-04)
- **Source type:** paper
- **Used in:** ch-29a §2.4, §7, Recipe

## Continued pretraining (§3.1, Table 5)
- Data: "Per-source upsampled pretraining corpus. Documents are separated using '<s>'", following Fu et al. (2024); 1B tokens, one epoch, LR 3e-5.
- "we separate individual documents using special characters rather than the reserved beginning and ending tokens
  ('<|begin_of_text|>' and '<|end_of_text|>'). Furthermore, we do not apply the cross-document attention mask (Gao et al., 2024)
  during continued pretraining, allowing the model to attend to the entire input sequence."
- Initialization Llama-3.1-8B-Instruct (RoPE base 5×10^5, scaling factor s = 8); YaRN with α = 1, β = 4; s = 128 / 256 / 512 for 1M / 2M / 4M.

## Instruction tuning (§3.2)
- 100K examples from general, math, and code SFT sets, responses refined with GPT-4o and GPT-4o-mini; batch 128, LR 5e-6.
- "our SFT blend exclusively comprises the short-context data described above, consisting of instances shorter than 8K tokens,
  without incorporating synthetic long-context instruction data ... We find that relying solely on short-context data is sufficient
  to achieve strong results in our setting".

## Results (evaluations run by the authors, including baselines)
- Table 1, RULER <1M: Llama-3.1-8B-Instruct 61.3; Gradient-1048k 77.8; ProLong-512k-Instruct 71.2; UltraLong-1M 79.1; 2M 78.2; 4M 78.0.
  InfiniteBench: 24.66; 28.60; 23.54; 32.14; 32.49; 30.38.
- Table 2, average of MMLU, MMLU-Pro, MATH, GSM-8K, HumanEval: Llama-3.1-8B-Instruct 61.45; Gradient 37.36; ProLong 40.81;
  UltraLong-1M 62.47; 2M 61.06; 4M 60.95.

## Ablations (Table 3, UltraLong-8B-1M, same 1B-token corpus)
| Variant | RULER <128K | <512K | <1M | LV-Eval <128K | <256K | InfiniteBench |
|---|---|---|---|---|---|---|
| UltraLong-8B-1M | 85.63 | 82.28 | 80.17 | 27.60 | 26.40 | 26.25 |
| w/o special separator | 85.47 | 81.63 | 79.15 | 26.06 | 24.85 | 22.75 |
| w/ NTK-aware scaling | 86.91 | 80.27 | 76.62 | 22.34 | 21.24 | 20.18 |

- The separator ablation removes the separator while keeping full attention; a with-mask versus without-mask run is not reported.
- Table 4: one-step extension to 1M (80.17 RULER <1M) beats 512K→1M two-step (77.52) at the same training cost.
