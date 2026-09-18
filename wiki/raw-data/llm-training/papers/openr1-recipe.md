<!-- scope: recipe ledger for the Hugging Face Open R1 releases — OpenR1-Math-220k generation and filtering, OpenR1-Qwen-7B, OlympicCoder-7B/32B, the math verification-filter ablation, OpenR1-Distill-7B, and the evaluation protocol
     deps: [[openr1]]
     see-also: [[deepseek-r1]], [[numina-math]]
-->

# Open R1: A fully open reproduction of DeepSeek-R1 — Recipe ledger
- **Parent card:** [[openr1]] (`papers/openr1.md`)
- **Sources:** github.com/huggingface/open-r1 README and `recipes/OpenR1-Distill-7B/sft/config_distill.yaml` at commit 1416fa0 (main on 2026-09-14); HF blogs "Open R1: Update #2" (2025-02-10) and "Open R1: Update #3" (2025-03-11); dataset cards open-r1/OpenR1-Math-220k and open-r1/Mixture-of-Thoughts; model card open-r1/OpenR1-Distill-7B.
- **Units:** "traces", "solutions", and "generations" are R1 completions; "problems" are prompts. Dataset row counts come from card metadata (`num_examples`). "Effective batch size" and "total_train_batch_size" are sequences per optimizer step.

## Data generation: OpenR1-Math-220k

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenR1-Math-220k (teacher DeepSeek-R1) | — | distill-SFT | prompts; source | 400k problems from NuminaMath 1.5 | Update #2 "Data generation"; dataset card "Dataset curation" | verified 2026-09-14 | no ablation reported |
| OpenR1-Math-220k | — | distill-SFT | temperature / top_p | "model card's recommended parameters"; values not printed | Update #2; dataset card | not reported | checked Update #2, dataset card, README; README's example R1 generation command on AI-MO/NuminaMath-TIR uses `--temperature 0.6` (example only, not tied to this dataset) |
| OpenR1-Math-220k | — | distill-SFT | max generation length | 16k tokens | Update #2; dataset card | verified 2026-09-14 | only 75% of problems solvable under 8k tokens; most others needed 16k |
| OpenR1-Math-220k | — | distill-SFT | samples per problem; total | 2, "in some cases, four"; 800k traces | Update #2; dataset card | verified 2026-09-14 | stated purpose: rejection sampling and DPO; no ablation reported |
| OpenR1-Math-220k | — | distill-SFT | inference engine; hardware | vLLM (15 generations/h/H100), then SGLang (25 solutions/h/H100); 512 H100 | Update #2 | verified 2026-09-14 | — |
| OpenR1-Math-220k | — | distill-SFT | daily throughput | "300k problem solutions per day on 512 H100s" vs "180k reasoning traces per day" | Update #2 "Data generation" vs bullet list in the same post | conflict | the dataset card repeats the 300k figure |
| OpenR1-Math-220k | — | distill-SFT | correctness filter | Math-Verify on extracted final answer; Llama-3.3-70B-Instruct judge on a subset of rejected problems (incomplete and empty-ground-truth samples removed first) | Update #2 "Data Filtering" | verified 2026-09-14 | 55% of problems pass Math-Verify; judge recovers 28,000 problems |
| OpenR1-Math-220k | — | distill-SFT | judge share | 12% of samples | dataset card "Dataset description" | verified 2026-09-14 | — |
| OpenR1-Math-220k | — | distill-SFT | released rows | `default` 93,733; `extended` 131,396; `all` 225,129 | dataset card metadata | verified 2026-09-14 | `default` scores best after SFT; `extended` lower (card) |
| OpenR1-Math-220k | — | distill-SFT | reward-model selection | Qwen2.5-Math-RM-72B top-1 among correct generations, scored on the answer without `<think>` | Update #2 | verified 2026-09-14 | no improvement over one random correct generation (training ablation, numbers not given) |

## SFT runs

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenR1-Qwen-7B | 7B | distill-SFT | base; data | Qwen2.5-Math-Instruct ("Qwen-7B-Math-Instruct" in the summary list); OpenR1-Math-220k `default` | Update #2 "Performance Comparison" | verified 2026-09-14 | dataset card: `default` > `extended` after SFT |
| OpenR1-Qwen-7B | 7B | distill-SFT | epochs; peak LR; schedule; warmup | 3; 5e-5; linear; 10% | Update #2 | verified 2026-09-14 | no ablation reported |
| OpenR1-Qwen-7B | 7B | long-context | context; RoPE | 4k → 32k by increasing RoPE frequency to 300k | Update #2 | verified 2026-09-14 | no ablation reported |
| OpenR1-Qwen-7B | 7B | distill-SFT | batch size; packing; loss masking | not reported | checked Update #2, dataset card | not reported | — |
| OlympicCoder-7B / -32B | 7B, 32B | distill-SFT | base; data | Qwen2.5 Coder Instruct 7B / 32B; codeforces-cots `solutions` and `solutions_w_editorials`, C++ only | Update #3 "Lessons learned" | verified 2026-09-14 | Lesson 3: editorial-conditioned traces did not beat problem-only traces |
| OlympicCoder-7B / -32B | 7B, 32B | distill-SFT | epochs; effective batch; peak LR; schedule | 10; 128; 4e-5; cosine decaying to 10% of peak | Update #3 | verified 2026-09-14 | Lesson 2: each LR doubling ≈ +10 LiveCodeBench points (figure) |
| OlympicCoder-7B | 7B | distill-SFT | context; parallelism | 32,768 tokens; DeepSpeed ZeRO-3 on one node of 8 × H100 | Update #3 Lesson 5 | verified 2026-09-14 | — |
| OlympicCoder-32B | 32B | distill-SFT | context; parallelism; optimizer | 22,528 tokens; FSDP; `paged_adamw_8bit` | Update #3 Lesson 5 | verified 2026-09-14 | OOM beyond 20k tokens "even on 16 nodes" otherwise; 9% of data truncated |
| OlympicCoder-7B | 7B | distill-SFT | packing | not used ("Combining all these lessons produced OlympicCoder-7B") | Update #3 Lessons 1 and 4 | verified 2026-09-14 | packing "consistently worse" on all ablated datasets (figure) |
| Filter ablation models | 7B | distill-SFT | base; data; epochs; context; schedule | "Qwen7B-Instruct"; 200k / 124k / 88.7k / 101k / 154k / 71.2k filtered sets; 1 epoch each; RoPE extension to 32k; cosine | Update #3 "Open R1 Math-Dataset update" | verified 2026-09-14 | recommendation: LorMV (Llama judge OR Math-Verify on re-parsed answers) |
| OpenR1-Distill-7B | 7B | distill-SFT | base | open-r1/Qwen2.5-Math-7B-RoPE-300k (Qwen2.5-Math-7B, RoPE base 10k → 300k) | config_distill.yaml@1416fa0; model card | verified 2026-09-14 | model card exp1–3: 100k / 300k / 500k no significant difference |
| OpenR1-Distill-7B | 7B | distill-SFT | data | Mixture-of-Thoughts `all`: 349,317 rows (math 93,733; code 83,070; science 172,514) | config@1416fa0; MoT card metadata | verified 2026-09-14 | MoT card and model card exp9–10: all three domains > math + code |
| OpenR1-Distill-7B | 7B | distill-SFT | epochs; peak LR; schedule; warmup | 5; 4.0e-5; cosine_with_min_lr, min_lr_rate 0.1; warmup_ratio 0.03 | config@1416fa0; model card | verified 2026-09-14 | model card exp4–6: LR 1e-5 / 2e-5 / 4e-5 per domain (figure) |
| OpenR1-Distill-7B | 7B | distill-SFT | batch | 2 per device × 8 grad-accum × 8 devices = 128 sequences | model card "Training hyperparameters"; config@1416fa0 | verified 2026-09-14 | — |
| OpenR1-Distill-7B | 7B | distill-SFT | max length; packing | 32,768; `packing: false` | config@1416fa0 | verified 2026-09-14 | model card exp7–8: packing vs no packing on math (figure) |
| OpenR1-Distill-7B | 7B | distill-SFT | optimizer; grad clip; precision; parallelism | Adam β = (0.9, 0.999), ε = 1e-8; max_grad_norm 0.2; bf16; DeepSpeed ZeRO-3, Liger kernel, gradient checkpointing | model card; config@1416fa0 | verified 2026-09-14 | no ablation reported |
| OpenR1-Distill-7B | 7B | distill-SFT | chat template; EOS | ChatML with a default "You are Open-R1…" system prompt that requests `<think>` Thought and Solution sections; `<|im_end|>` | config@1416fa0 | verified 2026-09-14 | no ablation reported |
| OpenR1-Distill-7B | 7B | eval-gate | checkpoint rule | `save_strategy: epoch`, `save_total_limit: 1`; ablations keep the best epoch on AIME 2024, GPQA Diamond, LiveCodeBench v4 | config@1416fa0; MoT card | verified 2026-09-14 | — |
| OpenR1-Distill-7B | 7B | distill-SFT | loss masking; weight decay | not reported | checked config@1416fa0, model card, README | not reported | — |

## Evaluation protocol

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| OpenR1-Distill-7B and R1-Distill baselines | 1.5B–70B | eval-gate | sampling | temperature 0.6, top_p 0.95, max_new_tokens 32,768, lighteval on vLLM | README "Evaluating models"; model card | verified 2026-09-14 | — |
| same | 1.5B–70B | eval-gate | samples per query for pass@1 | AIME 2024: 64; MATH-500: 4; GPQA Diamond: 8; LiveCodeBench: 16 | README "Reproducing Deepseek's evaluation results"; model card | verified 2026-09-14 | AIME 2024 has 30 problems, so few samples give high run-to-run variance (README) |
| training datasets | — | eval-gate | decontamination | 8-gram match against benchmark datasets, plus deduplication (`scripts/decontaminate.py`) | README "Data decontamination" | verified 2026-09-14 | method follows s1 |

Starting point for a small general-purpose run: for SFT of a 7B base model on about 350k R1 traces from mixed math, code, and science on one node of 8 × H100 with a 32,768-token context, the OpenR1-Distill-7B row values apply: 5 epochs, LR 4.0e-5 with cosine decay to 10% and 3% warmup, 128 sequences per step, no packing, gradient clipping at 0.2. These values produced the released checkpoint; the project did not report a separate tuning of epochs or clipping.
