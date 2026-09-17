---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: OLMo 2 report arXiv:2501.00656v3; Olmo 3 report arXiv:2512.13961v2; OLMo-core released scripts at commit 66f768b; SmolLM3 official blog (2025-07-08)
source_url: https://arxiv.org/abs/2501.00656 ; https://arxiv.org/abs/2512.13961 ; https://github.com/allenai/OLMo-core/tree/66f768b223171b5fde71cf67ccefa843ab9c51cb/src/scripts/official/OLMo3 ; https://huggingface.co/blog/smollm3
created_at: "2026-09-15"
---

# Excerpt: mid-training evidence from OLMo 2, Olmo 3, and SmolLM3 (read at the primary sources)

The library cards [[olmo-2]], [[olmo-3]], and [[smollm-3]] do not carry these tables or loci. Every value below was read in the primary source on 2026-09-15.

## OLMo 2 (arXiv:2501.00656v3)

- Stage 2 "mid-training (5–10% of training FLOPs), where we linearly decay the learning rate to zero over the remaining length of the run" (§2.3). 7B: cosine schedule over 5T tokens truncated after 4T (Table 3, §4.1).
- Souping: 7B anneals three times for 50B tokens with different data orders and averages them; 13B and 32B use three 100B runs plus one 300B run. Totals: 7B 4.05T tokens (3.90T pretraining stage), 13B 5.6T (5T), 32B 6.6T (6.06T) (§2.3).
- Dolmino 50B mix (Table 13, Mix %): filtered DCLM 47.2, decontaminated FLAN 16.6, StackExchange Q&A 2.45, peS2o 5.85, Wikipedia/Wikibooks 7.11, Dolmino Math 20.8. The math pool includes the GSM8K train split (Table 5).
- Table 9, 7B pretraining → pretraining & mid-training: MMLU 59.8 → 63.7; ARC-C 72.6 → 79.8; HellaSwag 81.3 → 83.8; WinoGrande 75.8 → 77.2; NQ 29.0 → 36.9; DROP 40.7 → 60.8; AGIEval 44.6 → 50.4; GSM8K 24.1 → 67.5; MMLU-Pro 27.4 → 31.0; TriviaQA 74.6 → 78.0. 1B: WinoGrande 67.8 → 66.5.
- Learning-rate finding (§4.1, Table 8): peak LR 3e-4 to 12e-4 for 300B tokens, then decay to zero over 50B or 100B tokens; OLMES averages vary by less than two points; "a higher learning rate does make mid-training more effective, but it does so by exactly the amount that the pretraining is worse."
- Microanneals (§4.4.2): ~50/50 math and general web data, LR driven linearly down; 19 microanneals totaling 130B tokens; evaluation on GSM* (200 of 1,319 GSM8K test questions; the other 1,119 are held out, footnote 6). Table 12, experiment 3 (baseline GSM* 28.5, MMLU 59.8): TinyGSM-Inline (code answers) GSM* 25.0, MMLU 60.4; TinyGSM-MIND (natural-language rewrite) 65.5, 61.4; 2× TinyGSM-MIND 70.0, 62.1.
- Table 14: souping three 50B runs equals or exceeds the best single checkpoint on OLMES, OLMES-Gen and MMLU for all six candidate mixes; mix E GSM* falls from 60.5 (best single) to 43.0 (soup).

## Olmo 3 (arXiv:2512.13961v2)

- Dolma 3 Dolmino Mix: 99.95B tokens sampled from a 2.19T pool (Table 5). Shares: Common Crawl HQ 22.5%; Dolmino Math 10.7%; StackEdu (FIM) 10.0%; CraneCode 10.0%; Reddit To Flashcards 5.9%; Nemotron Synth QA 5.0%; Dolmino 1 Flan 5.0%; olmOCR science PDFs HQ 5.0%; STEM-heavy crawl 5.0%; CraneMath 5.63%; Wiki To RCQA 3.0%; QWQ reasoning traces 1.87%; General reasoning mix 1.87%; MegaMatt 1.73%; Llama Nemotron reasoning traces 1.25%; OpenThoughts2 reasoning traces 1.25%; Tulu 3 SFT 1.1%; TinyMATH Mind 0.9%; OMR rewrite FullThoughts 0.85%; code meta-reasoning 0.46%; math meta-reasoning 0.38%; Gemini reasoning traces 0.25%; TinyMATH PoT 0.24%; program-verifiable 0.16%.
- Microanneal (§3.5.1): 5B tokens of the target dataset + 5B web tokens, compared with a 10B web-only microanneal. Integration tests: full 100B candidate mixes, evaluated on OlmoBaseEval and after SFT.
- Table 6 (OlmoBaseEval avg / Math / SFT avg): Round 1 49.7 / 47.4 / 35.2; Round 3 50.7 / 48.7 / 35.3; Round 5 53.1 / 57.1 / 37.3.
- Table 7 (MC STEM, MC Non-STEM, GenQA, Math, Code, FIM): Gen-QA mix 66.3, 78.1, 72.5, 27.5, 11.9, 0.1; math-code-thinking mix 62.5, 69.6, 65.9, 60.8, 35.6, 37.7; final Round 5 mix 66.4, 77.4, 73.1, 57.3, 31.2, 31.7.
- Table 9 (5B microanneal, web-only vs meta-reasoning + program-verifiable): MMLU 55.2 vs 53.7; GenQA 53.7 vs 52.9; GSM8K 18.4 vs 26.8; HumanEval 7.9 vs 19.5.
- Table 10 (100B integration test): without thinking traces/instruction data avg 48.8, Math 43.1; full mix avg 50.7, Math 48.7.
- Special tokens (§3.5.4): Tulu3-SFT microanneal with chat special tokens: GSM8K 49.43 → 0 and CruxEval 32.89 → 18.91; chat template with ordinary text in place of special tokens: 46.02 and 29.65. Final choice: newline-based formatting without special tokens.
- Decontamination (§3.5.3–3.5.4): decon removes matches against all splits of all OLMES benchmarks; over 60,000 DROP training examples removed from sources such as Flan; complete GSM8K leakage was detected, but GSM8K was higher with decontaminated data.
- 32B: two 100B midtraining runs with different seeds merged; Math cluster +2.9 and +1.6 over the two runs (§3.5.4). 7B midtrained checkpoint is a single run (footnote 24).
- Long-context extension (§3.6, §3.6.3, Table 11, Table 13): 8,192 → 65,536 tokens; 34% long-context data + 66% Dolmino short data; 50B tokens (7B) or 100B (32B); YaRN on full-attention layers only; in a 10B-token extension, a 66% long / 34% short mix dropped a subset of OlmoBaseEval by 2.5 points and a 34% / 66% mix by 0.8 points. Table 13, 7B Stage 2 → Stage 3: Math 59.8 → 54.4; Code 31.9 → 30.6; GenQA 71.3 → 72.5; GenXL 49.1 → 43.6; MMLU 66.9 → 66.9. 32B Stage 2 soup → Stage 3: Math 69.7 → 61.4.
- Placement across families, as characterized by the Olmo 3 authors: "Llama 3.1 models apply long-context extension prior to midtraining, Qwen 2.5 and 3 perform it afterwards, and GLM 4.5 applies extension only after supervised finetuning" (§3.6).

## OLMo-core released scripts (commit 66f768b)

`src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`:

```python
40  DEFAULT_SEQUENCE_LENGTH = 8192
41  GLOBAL_BATCH_SIZE = 2**21  # ~2M tokens
42  MAX_TOKENS = 100_000_000_000  # 100B
43  LR = 0.00020712352850360292
...
72          optim=SkipStepAdamWConfig(
73              lr=LR,
74              weight_decay=0.1,
75              betas=(0.9, 0.95),
...
80          scheduler=LinearWithWarmup(warmup=0, alpha_f=0.0),
...
103             load_optim_state=True,
```

`src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py`:

```python
37  DEFAULT_SEQUENCE_LENGTH = 65536
38  GLOBAL_BATCH_SIZE = 65536 * 64  # ~4M tokens
49      ).with_rope_scaling(
50          YaRNRoPEScalingConfig(factor=8, beta_fast=32, beta_slow=1, old_context_len=8192)
81          scheduler=LinearWithWarmup(warmup=200, alpha_f=0.0),
89          cp_config=TransformerContextParallelConfig.llama3(degree=8, head_stride=4),
100             load_path="https://olmo-checkpoints.org/ai2-llm/Olmo-3-1025-7B/stage2/step47684/",
```

`src/olmo_core/optim/scheduler.py` L350–384: `LinearWithWarmup` sets `eta_min = initial_lr * self.alpha_f`, so `alpha_f=0.0` decays to zero.

## SmolLM3 official blog (2025-07-08)

- Pretraining: WSD, 2,000 warmup steps, "linear decay to 0 in the final 10% training steps"; stage 3 decay phase 10T → 11.1T tokens with web 63%, code 24%, math 13% ("Training Configuration", "Stage 3").
- Long-context: 100B tokens; 4k → 32k with RoPE θ 1.5M (50B), then 32k → 64k with θ 5M (50B) ("Long Context extension").
- Reasoning mid-training: 35B tokens from OpenThoughts3-1.2M and a Llama-Nemotron-Post-Training-Dataset-v1.1 subset with R1 traces, ChatML template, 4 epochs (~140B tokens) ("Reasoning Mid-training").
- RULER degraded after post-training; the team "traced this degradation back to the reasoning mid-training stage"; a linear merge of 0.9 × APO soup + 0.1 × mid-training checkpoint recovered the base model's RULER score up to 128k ("Model Merging").

## Used in

ch-32 §1–§3, §6, §7, Recipe, figure panel A.
