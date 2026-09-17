<!-- excerpt for: ch-32e
     source: Olmo Team, "Olmo 3", arXiv:2512.13961v2 (2026-04-14; v1 2025-12)
     scope: Olmo 3 Base stages 2 (midtraining, Dolma 3 Dolmino Mix) and 3 (long-context extension, Dolma 3 Longmino Mix); Tables 5-13, 33-35
     library card: [[olmo-3]] (predates the 2026-09 verification; see last section)
-->

# Olmo 3 Base midtraining and long-context extension — verified extract

Checked on 2026-09-15 against the arXiv v2 PDF text.

## Stage settings (Table 35; architecture Table 33)
| Setting | 7B midtrain | 7B long-context | 32B midtrain | 32B long-context |
|---|---|---|---|---|
| LR schedule | linear decay | linear decay | linear decay | linear decay |
| Warmup from 0 | 0 steps | 200 steps | 0 steps | 200 steps |
| Peak LR | 2.074e-4 | 2.074e-4 | 2.071e-4 | 2.071e-4 |
| Final LR | 0 | 0 | 0 | 0 |
| Batch (instances × length) | 256 × 8,192 | 64 × 65,536 | 512 × 8,192 | 128 × 65,536 |
| Batch (tokens) | 2,097,152 | 4,194,304 | 4,194,304 | 8,388,608 |
| Total tokens | 100B | 50B | 100B (twice) | 100B |

Pretraining: 7B 5.93T tokens, peak LR 3.0e-4, 4,194,304 tokens per batch; 32B 5.93T cosine truncated at 5.5T, peak 6.0e-4 (Table 35). RoPE θ = 5 × 10^5; sliding-window attention on 3/4 of layers with a 4,096-token window; YaRN on full-attention layers (Table 33).

## Midtraining (§3.5)
- Dolma 3 Dolmino Mix: 99.95B tokens sampled from a 2.19T pool (Table 5). Largest shares: Common Crawl HQ 22.5%, Dolmino Math 10.7%, StackEdu (FIM) 10.0%, CraneCode 10.0%, Reddit To Flashcards 5.9%; thinking traces and instruction data are listed as separate rows.
- Microanneal: 5B target tokens + 5B web tokens, compared against a 10B web-only microanneal (§3.5.1). Integration tests: full 100B anneals, each followed by SFT and post-training evaluation (§3.5.1).
- Table 6 (candidate mixes on earlier checkpoints): OlmoBaseEval average Round 1 49.7, Round 3 50.7, Round 5 53.1; SFT-experiment average 35.2, 35.3, 37.3.
- Table 7 (domain-skewed 100B anneals): Gen-QA mix Math 27.5 / Code 11.9 / GenQA 72.5; math-code-thinking mix Math 60.8 / Code 35.6 / GenQA 65.9 / MC Non-STEM 69.6; final mix Math 57.3 / Code 31.2 / GenQA 73.1 / MC Non-STEM 77.4.
- Table 10: without thinking traces and instruction data average 48.8 (Math 43.1); full mix 50.7 (Math 48.7), total tokens held constant.
- Special chat tokens in midtraining data: GSM8K 49.43 → 0 and CruxEval 32.89 → 18.91 in microanneals on Tulu3-SFT data; a chat template with ordinary text instead of special tokens gave 46.02 and 29.65 (§3.5.4).
- 32B: two midtraining runs with different data-order seeds were merged; Math cluster +2.9 and +1.6 over the two runs, about +1 MMLU (§3.5.4). 7B: initial experiments showed no similar gain, so the 7B midtrained checkpoint is one run (footnote 24).

## Long-context extension (§3.6)
- Context 8,192 → 65,536 tokens. Mix: 34% long-context data, 66% short data sampled from Dolmino Mix; 50B (7B) and 100B (32B) tokens (§3.6). Table 11: long rows total 16.95B of 50.0B; "Midtraining data mix" 33.0B (66.1%).
- Early 10B-token extension: 66% long / 34% short dropped a subset of OlmoBaseEval by 2.5 points; 34% long / 66% short dropped it by 0.8 points (§3.6.3).
- Recipe components (Figure 13, RULER only, values shown as bars): YaRN on full-attention layers only; olmOCR PDFs over ProLong data; synthetic CWE/REX augmentation; best-fit document packing with intra-document masking; longer extension budgets (1B to 100B).
- gzip filter removes the 20% most and 20% least compressible text (§3.6.1). 8-way context parallelism (§3.6.4).
- 32B final long-context checkpoint = average of steps 10,000, 11,000 and 11,921 (§3.6.4).
- RULER is the development suite and HELMET the held-out suite; footnote 25 notes some overlap (§3.6). Table 12, Olmo 3 32B: RULER 4K 96.10 / 32K 86.22 / 65K 79.70; HELMET 8K 52.11 / 65K 43.15.
- Table 13 (Stage 2 → Stage 3): 7B Math 59.8 → 54.4, GenQA 71.3 → 72.5, MMLU 66.9 → 66.9; 32B (soup → Stage 3) Math 69.7 → 61.4, Minerva 46.9 → 42.9, MMLU 76.9 → 76.2.
- The report surveys other recipes' extension budgets: SmolLM3 100B, GLM 4.5 100B, DeepSeek V3 123B, Llama 3.1 800B, DeepSeek V3.1 840B; it also states that GLM 4.5 "applies extension only after supervised finetuning" (§3.6), which disagrees with GLM-4.5's own Figure 3 and §2.3 (128K mid-training stage).

## Disagreements with the library card [[olmo-3]]
- "Longmino about 50B" → 50B for 7B, 100B for 32B (§2.1, Table 35).
- "Mid-training used 128 H100 GPUs" → 7B only; 32B ran two midtraining runs on 512 GPUs each (Table 34; §2 runtime).
