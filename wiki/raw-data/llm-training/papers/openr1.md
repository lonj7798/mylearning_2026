<!-- scope: Hugging Face Open R1 project (Jan–May 2025) — repository, plan, and the R1-distilled releases OpenR1-Math-220k, CodeForces-CoTs, Mixture-of-Thoughts, OpenR1-Qwen-7B, OlympicCoder, and OpenR1-Distill-7B
     deps: [[deepseek-r1]]
     see-also: [[bespoke-stratos]], [[sky-t1]], [[open-thoughts]], [[numina-math]], [[packed-vs-unpacked-ablation]], [[openr1-recipe]]
-->

# Open R1: A fully open reproduction of DeepSeek-R1
- **Core Insight:** Supervised fine-tuning of Qwen2.5-Math-7B (RoPE base frequency extended to 300k) on Mixture-of-Thoughts, 350k DeepSeek-R1 reasoning traces for math, code, and science, gives OpenR1-Distill-7B 52.7 AIME 2024, 89.0 MATH-500, 52.8 GPQA Diamond, and 39.4 LiveCodeBench v5, against 51.3, 93.5, 52.4, and 37.4 for DeepSeek-R1-Distill-Qwen-7B from the same base (README "SFT distillation"; OpenR1-Distill-7B model card).
- **Guideline:** When distilling R1 traces into a 7B model with SFT, disable sequence packing and include learning rates up to 4e-5 in the sweep, because in the project's OlympicCoder SFT ablations on Qwen2.5-Coder-Instruct, packing was worse on every ablated dataset and each doubling of the learning rate up to 4e-5 added almost 10 LiveCodeBench points (Update #3, Lessons 1–2; results shown only as figures).
- **Authors:** Hugging Face (repository citation). Launch blog: Elie Bakouch, Leandro von Werra, Lewis Tunstall. Update #2: Loubna Ben Allal, Lewis Tunstall, Anton Lozhkov, Elie Bakouch, Guilherme Penedo, Hynek Kydlicek, et al. Update #3: Guilherme Penedo, Lewis Tunstall, Anton Lozhkov, Hynek Kydlicek, Edward Beeching, Loubna Ben Allal, et al.
- **Year:** 2025 (repository and launch blog 2025-01-28; Update #2 2025-02-10; Update #3 2025-03-11; step 1 declared complete 2025-05-26)
- **URL:** https://github.com/huggingface/open-r1 (README at commit 1416fa0) ; https://huggingface.co/blog/open-r1
- **Source type:** released config/code (repository), with official blogs and dataset/model cards from the same team
- **Relevant topics:** R1 replication, reasoning distillation, long-CoT SFT, answer verification, data filtering, sequence packing, evaluation variance, GRPO tooling

## Summary
Open R1 is Hugging Face's project to rebuild the parts of the DeepSeek-R1 pipeline that were not released: the reasoning datasets and the training code. The plan has three steps: (1) replicate the R1-Distill models by distilling a reasoning corpus from DeepSeek-R1; (2) replicate the pure-RL R1-Zero pipeline, which needs new large-scale math, reasoning, and code datasets; (3) show a base model → SFT → RL multi-stage pipeline (launch blog; README "Plan of attack"). The repository provides `sft.py`, `grpo.py`, and `generate.py` (Distilabel), lighteval evaluation commands, and an 8-gram decontamination script (README). Released artifacts include OpenR1-Math-220k (Update #2), CodeForces-CoTs, IOI benchmark data, and OlympicCoder-7B/32B (Update #3), and Mixture-of-Thoughts with OpenR1-Distill-7B, which the README calls the completion of step 1.

## Key Contributions
- OpenR1-Math-220k: R1 traces for NuminaMath 1.5 problems, verified with Math-Verify plus an LLM judge, Apache 2.0 (Update #2; dataset card).
- CodeForces-CoTs (close to 100k R1 samples in C++ and Python for more than 10k problems) and processed IOI 2020–2024 problems with test cases and grading code (Update #3).
- Published SFT ablations on R1 traces: packing, learning rate, editorial conditioning, `<think>` prefill, verification filters, domain mixture, RoPE base (Update #3; Mixture-of-Thoughts card; OpenR1-Distill-7B model card).
- A pinned SFT config for OpenR1-Distill-7B and an evaluation protocol with a stated number of samples per benchmark (README; `recipes/OpenR1-Distill-7B/sft/config_distill.yaml`).
- Negative results reported as such: reward-model selection among correct traces gave no gain (Update #2); editorials in the prompt did not help (Update #3).

## Key Figures/Tables to Study
- Update #2 table (OpenR1-Qwen-7B vs DeepSeek-Distill-Qwen-7B vs OpenThinker-7B); Update #3 Lesson 1–3 figures and the verification-filter results; Mixture-of-Thoughts card figures (per-domain vs mixture); README AIME/MATH-500/GPQA/LiveCodeBench reproduction tables.

## Technical Details
**OpenR1-Math-220k (Update #2; dataset card)**
- DeepSeek-R1 generates solutions for 400k NuminaMath 1.5 problems with the R1 model card's recommended generation parameters and the suffix "Please reason step by step, and put your final answer within \boxed{}."; the blog and card do not print the temperature.
- Limit 16k tokens per generation, because only 75% of problems were solved within 8k tokens and most of the rest needed the full 16k.
- Two solutions per problem, four for some problems; the stated purposes are rejection sampling and preference optimization such as DPO. Total: 800k traces.
- Throughput: vLLM 15 generations per hour per H100; SGLang 25 solutions per hour per H100, "300k problem solutions per day on 512 H100s". The same blog also states "generating 180k reasoning traces per day".
- Filtering: Math-Verify compares extracted final answers with ground truth; 55% of problems have at least one correct answer. Llama-3.3-70B-Instruct judges a subset of rejected problems, after removing incomplete samples and empty ground truths, and recovers 28,000 problems. The card states the judge covers 12% of samples and each problem keeps at least one correct trace.
- Splits: `default` 94k problems (93,733 rows), best after SFT; `extended` 131k (131,396 rows) adds sources such as cn_k12 and scores lower after SFT, which the authors attribute to easier questions (dataset card; card metadata).
- Reward-model filter: choosing the top-1 correct generation with Qwen2.5-Math-RM-72B did not improve over a random correct generation (Update #2).
- OpenR1-Qwen-7B (Qwen2.5-Math-Instruct, 3 epochs on `default`): MATH-500 90.6, AIME24 36.7, AIME25 40.0; DeepSeek-Distill-Qwen-7B 91.6 / 43.3 / 40; OpenThinker-7B 89.6 / 30.0 / 33.3 (Update #2 table).

**CodeForces-CoTs and OlympicCoder (Update #3)**
- Seven R1 solutions that passed all public CodeForces tests all failed the full test set on the platform; CodeForces caps displayed tests at about 500 characters, and the authors conclude that such datasets are not fully verifiable.
- IOI 2024: 50 submissions per subtask with a round-robin selection similar to OpenAI's o1-ioi strategy; OlympicCoder-32B outperforms o1-mini and DeepSeek-R1 under the 50-submission limit (figure).
- Lessons: packing hurt on all ablated datasets; LR 4e-5 beat 2e-5; R1 traces conditioned on editorials did not beat traces from problem statements alone; without a `<think>` prefill, out-of-domain prompts reverted to the base instruct model's behavior; the 32B model needed FSDP with `paged_adamw_8bit` to reach 22,528 tokens.

**Verification-filter ablation (Update #3 "Open R1 Math-Dataset update")**
- Six SFT sets from a random 200k pool: no filter 200k; Llama judge 124k; Math-Verify on `answer` 88.7k; Math-Verify on re-parsed answers 101k; union (LorMV) 154k; intersection (LandMV) 71.2k; one epoch each, Qwen 7B Instruct, 32k context.
- In the first 40 steps, stricter filters help: MATH-500 0.61 without filtering vs 0.72 for LandMV; the gap shrinks later, and the unfiltered set stays competitive in longer training. The authors recommend LorMV.

**Mixture-of-Thoughts and OpenR1-Distill-7B (card; model card; config at 1416fa0)**
- Composition: math 93.7k (`default` of OpenR1-Math-220k), code 83.1k (`solutions`, `solutions_w_editorials` of codeforces-cots), science 173k (subset of the Llama-Nemotron post-training science split with no Qwen prompt pre-processing); total 349,317 rows.
- Mixture method: optimize each domain independently, then combine, following Phi-4-reasoning; evaluate AIME 2024, GPQA Diamond, LiveCodeBench v4 every epoch and keep the best checkpoint. Training on all domains together gave the best results.
- Math: `default` beat `extended`, and both together were best, but only `default` was used to limit dataset size. RoPE base 100k, 300k, 500k showed no significant difference; 300k was used.
- Evaluation (README): temperature 0.6, top_p 0.95, 32,768 max new tokens; pass@1 estimated from 64 (AIME 2024), 4 (MATH-500), 8 (GPQA Diamond), 16 (LiveCodeBench) samples per query, because AIME 2024 has 30 problems and single runs vary.

## Recipe ledger
The full ledger (OpenR1-Math-220k generation, OpenR1-Qwen-7B, OlympicCoder-7B/32B, the verification ablation, and OpenR1-Distill-7B) is in [[openr1-recipe]]. OpenR1-Distill-7B summary: 5 epochs, LR 4.0e-5, cosine with min_lr_rate 0.1, warmup ratio 0.03, 128 sequences per step, max length 32,768, packing off, max_grad_norm 0.2, DeepSpeed ZeRO-3 on 8 × H100 (config_distill.yaml@1416fa0; model card).

## Findings relevant to generality, negative feedback, long context, distillation
- Generality: all-domain training beat single-domain training (Mixture-of-Thoughts card). Easier cn_k12 questions lowered SFT results (dataset card). Without a `<think>` prefill, out-of-domain prompts did not trigger long reasoning; the authors recommend enforcing the prefill in the chat template (Update #3 Lesson 4).
- Measurement: AIME 2024 has 30 problems, so pass@1 is averaged over 64 samples; the README reproduces DeepSeek's distilled-model scores "within ~1-3 standard deviations". The README decontaminates datasets with 8-gram matching following s1.
- Negative samples: failed traces are discarded (negative marginal value). Keeping unverified traces as positives reduced early MATH-500 scores but was competitive later (Update #3). No negative-gradient method is evaluated in these sources.
- Long context: 16k generation limit (dataset card); 32k SFT via RoPE base 300k (Update #2; model card); 20% of CodeForces-CoTs traces exceed 20k tokens, and 9% are truncated at 22,528 (Update #3).
- Distillation: every released model is SFT on DeepSeek-R1 traces; no GRPO-trained model or GRPO result appears in the checked sources. GRPO support lives in TRL, and Update #3 suggests 2–4 reuses per generation (μ).

## Connections
- [[deepseek-r1]] — teacher model and the pipeline being reproduced.
- [[numina-math]] — NuminaMath 1.5 supplies the OpenR1-Math-220k problems.
- [[open-thoughts]] — parallel open reasoning-data project; uses OpenR1-Math data and OpenR1-Distill-7B as a baseline.
- [[bespoke-stratos]], [[sky-t1]] — earlier open R1/QwQ distillation datasets listed in Update #1.
- [[s1]], [[limo]] — small curated sets discussed in Update #2; s1 is the source of the decontamination method.
- [[packed-vs-unpacked-ablation]] — packing effects; compare with Update #3 Lesson 1.
- [[grpo]] — algorithm supported by `grpo.py`.

## Verification
- Checked on 2026-09-14 against: github.com/huggingface/open-r1 README and recipes/OpenR1-Distill-7B/sft/config_distill.yaml at 1416fa0; HF blogs open-r1 (2025-01-28), update-1, update-2, update-3, update-4; dataset cards open-r1/OpenR1-Math-220k and open-r1/Mixture-of-Thoughts; model card open-r1/OpenR1-Distill-7B.
- Corrections to the previous card version: title "Open-R1: An open-source reproduction of DeepSeek-R1" → "Open R1: A fully open reproduction of DeepSeek-R1" (README citation); authors list containing Nathan Lambert → blog author lists above (he is not listed on any checked page); "220K problems × 2 traces ≈ 440K samples" → 400k problems × 2 (some 4) = 800k traces, 220k problems kept, released splits 93,733 and 131,396 rows; "2 traces at temperature 0.6, up to 8×" → 2, sometimes 4, with R1 model-card parameters (no temperature printed); "seed: NuminaMath cn_k12, olympiads, aops_forum, amc_aime, orca_math plus AIME/AMC archives" → NuminaMath 1.5, with cn_k12 only in `extended`; "HF H100 cluster + API mix" → local generation on 512 H100s "instead of relying on an API"; "OpenR1-Qwen-7B MATH ~80%, AIME24 ~40%" → MATH-500 90.6, AIME24 36.7, AIME25 40.0; "step 3: full SFT+RL+SFT+RL pipeline" → "base model → SFT → RL via multi-stage training"; "math-only; other tracks in progress" → CodeForces-CoTs and Mixture-of-Thoughts (math, code, science) were released; filtering "regex boxed extraction + format rejection" → Math-Verify plus Llama-3.3-70B-Instruct judge, with incomplete samples removed before judging.
- Removed as unsupported by the source: ~$10K inference cost; average ~5K tokens, tail to 30K, median ~5K / mean ~7K, ~10% over 15K; ~700-token and ~3K-token comparisons; ~20% verifier failure rate; GRPO on a 40K subset with KL penalty; +3–5 AIME from GRPO; "2 vs 1 traces gives marginal gain" ablation; "Math-Verify does not catch semantic drift (wrong question answered correctly)"; license-attribution and "R1 dependency" risk statements; "SymPy-based" description of Math-Verify (not stated in the checked pages).
- Not reported by the source: generation temperature for OpenR1-Math-220k; batch size and packing for OpenR1-Qwen-7B; loss masking settings; trace-length statistics for OpenR1-Math-220k; any GRPO training outcome.
