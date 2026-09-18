<!-- scope: Magistral paper (Mistral AI, arXiv:2506.10910): Magistral Medium trained with RL only from Mistral Medium 3; Magistral Small cold-started by SFT on Magistral Medium traces then RL; modified GRPO without KL, asynchronous RL infrastructure, data filtering, ablations and failed approaches
     deps: [[grpo]]
     see-also: [[magistral-recipe]], [[deepseek-r1]], [[dr-grpo]], [[llama-4]], [[kimi-k1-5]]
-->

# Magistral
- **Core Insight:** RL alone on Mistral Medium 3 raises AIME'24 pass@1 from 26.8 to 73.6 (maj@64 43.4 to 90.0) (Table 2), and for a 24B model RL on top of SFT on Magistral Medium traces reaches 70.7 AIME'24 pass@1, compared with 65.4 for SFT alone and 65.8 for RL alone (Table 3).
- **Guideline:** When a stronger teacher model exists, cold-start a smaller model with SFT on the teacher's correct reasoning traces and then run RL, because the paper's 24B comparison gives the best pass@1 on AIME'24, AIME'25, MATH-500, and LiveCodeBench v5/v6 for SFT + RL (Table 3); without a teacher, RL alone on the 24B model matched SFT on AIME'24 pass@1 (65.8 vs 65.4) but was lower on LiveCodeBench v5 (46.4 vs 52.2) (Table 3).
- **Authors:** Mistral AI; core contributors listed alphabetically: Abhinav Rastogi, Albert Q. Jiang, Andy Lo, Gabrielle Berrada, Guillaume Lample, Jason Rute, et al. ("Core contributors")
- **Year:** 2025 (arXiv v1 2025-06; no venue)
- **URL:** https://arxiv.org/abs/2506.10910
- **Source type:** official technical report
- **Relevant topics:** GRPO modifications, KL removal, Clip-Higher, zero-advantage filtering, reward shaping, language-consistency reward, asynchronous RL, data difficulty filtering, distillation vs RL, cross-domain generalization, multimodal retention

## Abstract
The paper introduces Magistral, Mistral's first reasoning model, and the RL pipeline used to train it. The authors use only their own models and infrastructure, not implementations or RL traces distilled from prior models. They report a stack for pure RL training of LLMs, a method to force the reasoning language to match the user's language, and the finding that RL on text data alone maintains most of the initial checkpoint's capabilities. RL on text maintains or improves multimodal understanding, instruction following, and function calling. Magistral Medium is trained on top of Mistral Medium 3 with RL alone. Magistral Small (Apache 2.0) additionally uses cold-start data from Magistral Medium (Abstract).

## Key Contributions
- A GRPO variant with no KL term, token-sum loss normalization, minibatch advantage normalization, Clip-Higher, and removal of zero-advantage groups (§2.1).
- A reward with format, correctness, length, and language-consistency terms (§2.2).
- An asynchronous generator/trainer/verifier system that updates generator weights over NCCL without stopping in-flight generations (§3).
- An SFT-then-RL comparison at 24B showing gains from RL over the distillation baseline, which the authors present as contradicting DeepSeek-R1's small-model observation (§5.3, §6.2).
- Reports of unsuccessful approaches: proportional code rewards and entropy bonuses (§7.4).

## Key Figures/Tables to Study
- Table 2: Magistral Medium vs Mistral Medium 3 and DeepSeek-v3 / R1-Zero / R1.
- Table 3: 24B SFT vs RL-only vs SFT + RL, including maj@64.
- Table 5: math-only and code-only RL evaluated on both domains.
- Fig. 6 (batch and minibatch size), Fig. 12 (ε_high vs entropy bonus), Fig. 13 (RL after SFT on open-source traces).

## Technical Details
Full settings with loci: [[magistral-recipe]].
- **Objective (§2.1).** GRPO computes an advantage from G generations per prompt instead of a critic. Magistral uses Â_i = r_i − μ, where r_i is the reward of generation i and μ the group mean, then normalizes Â_i by the mean and standard deviation of advantages in the minibatch. The loss sums token losses over all generations and divides by the total generated length Σ|o_i|. The ratio is clipped to [1 − ε_low, 1 + ε_high]; ε_high was adjusted between 0.26 and 0.28 during training. ε_low is not reported. Groups whose generations all have the same reward are removed.
- **KL (§2.1, §7.4.2).** The KL penalty is removed because "the policy diverges substantially regardless" and the reference-model compute is "unjustified". An exponential moving average of weights as KL reference was tried; adjusting ε_high manually was simpler.
- **Reward (§2.2).** Missing or malformed `<think>` tags, missing `\boxed{}` (math), or missing fenced code block (code) gives 0. A well-formatted response gets 0.1, plus 0.9 if correct (math: parsers + SymPy against the reference; code: 20 random tests, 4 s and 300 MB each, C++20 compile timeout 10 s). Length penalty (Eq. 1): 0 up to l_max − l_cache, linear to −0.1 at l_max, −0.1 beyond. Language consistency: +0.1 when a fastText classifier assigns the same language to problem, thoughts, and answer; 10% of English problems were translated into 6 languages.
- **System prompt (§2.2.4, Fig. 2).** RL is "quite sensitive" to the system prompt; the phrase "Be as casual and as long as you want" increases entropy.
- **Infrastructure (§3).** Trainers, generators, and verifiers run as separate workers. Weight broadcast over NCCL takes under 5 seconds; in-flight sequences keep their key-value cache. The longest completions can take up to 5 times longer than the shortest. A blocking queue with a size limit bounds off-policy degree. Greedy collation of sequences into microbatches reduces padding by 19%.
- **Data (§4).** Math: 699k → 501k (format filter) → 38k (difficulty filter) (Table 1). Difficulty filtering uses 16 samples per problem, first from Mistral Large 2, then from a 24B RL-trained grader; problems where most samples agree on an answer that differs from the reference are removed. Code: tests with insufficient agreement across solutions are discarded; tests with agreement but no passing solution are rewritten to the most common output; missing tests are generated; 35k problems (§4.2).
- **Magistral Medium schedule (§5.2).** Data difficulty is raised as performance increases; l_max − l_cache grows 16k → 24k → 32k; batch size falls 8k → 4k → 2k to limit KV-cache memory.
- **Magistral Small (§5.3).** SFT of Mistral Small 3 Instruct (24B) for 4 epochs on correct Magistral Medium RL traces plus Magistral Medium responses to OpenThoughts and OpenR1-code prompts, with 10% general instruction data; best AIME'24 checkpoint selected. RL: batch 2048 sequences, l_max − l_cache 32k, temperature 1.0, ε_high 0.3 because the cold-started model had lower entropy.
- **Results.** Magistral Medium (Table 2): AIME'25 21.2 → 64.9 pass@1; GPQA 59.6 → 70.8; LiveCodeBench v5 29.1 → 59.4; Humanity's Last Exam (text) 4.4 → 9.0. The paper calls the AIME'24 change "a nearly 50% boost"; 73.6 − 26.8 = 46.8 points (derived). Evaluation: temperature 0.7, top-p 1.0 (math, GPQA) or 0.95 (code), 40k max tokens for AIME and LiveCodeBench (§5.1).
- **Table 3 (24B, SFT / RL-only / SFT + RL).** AIME'24 pass@1 65.4 / 65.8 / 70.7; AIME'24 maj@64 90.0 / 86.7 / 83.3; AIME'25 pass@1 55.6 / 51.9 / 62.8; GPQA 63.4 / 68.8 / 68.2; LiveCodeBench v5 52.2 / 46.4 / 55.8.
- **Batch ablation (§6.3, Fig. 6).** 3B model from Ministral 3B, math-only GRPO, n_async = 4096: reward is similar across n_batch when n_batch = n_minibatch, degrades with more than 2 minibatches per batch, and is less stable at n_batch ≤ 1024. Final training kept n_async/n_batch ≤ 2 and n_batch = n_minibatch.
- **Advantage normalization (§6.4, Fig. 7).** Minibatch, group, and no normalization give no significant difference in evaluation or length growth.

## Findings relevant to generality, negative feedback, long context, distillation
- **Generality: multimodal (§7.2, Fig. 10).** Text-only RL on multimodal starting checkpoints: MMMU +5% to 70%, MMMU-Pro-Standard +4.4% to 57.9%, MMMU-Pro-Vision +12% to 52.1%; "no performance regression across most benchmarks". Result (single study).
- **Generality: other capabilities (§7.3, Table 6).** Mistral Medium 3 → Magistral Medium: internal function-calling benchmark 87.2 → 87.4; internal IFEval version 86.8 → 87.4. The paper states these scores are not comparable with public scores.
- **Generality: cross-domain (§6.1, Table 5).** 24B model from a checkpoint at AIME'24 32.2 / LiveCodeBench v5 22.7: math-only RL gives 62.5 / 38.3; code-only RL gives 49.7 / 42.7.
- **Generality: multilingual (§5.4, Table 4).** Magistral Medium AIME'24 pass@1 is 73.6 in English and 63.7-69.3 in six translated versions (4.3-9.9% lower), a drop the authors call "roughly similar to that of the base model".
- **Coverage measurement.** In Table 3, SFT + RL has the highest AIME'24 pass@1 (70.7) but the lowest AIME'24 maj@64 (83.3 vs 90.0 for SFT). The paper does not discuss this.
- **Negative feedback.** Zero-advantage groups are discarded (§2.1). Within a group, below-mean generations receive negative advantage and are used as gradient (follows from Â_i = r_i − μ, §2.1). A proportional code reward (fraction of tests passed) discarded three times less data but gave 2% lower LiveCodeBench and slower length growth (24B, 250 steps; §7.4.1, Fig. 11); the authors suggest partial rewards may give "false signal to incorrect solutions" (Interpretation). An entropy bonus lowered entropy on math-only data and made it explode on math + code data at the same coefficient, while higher ε_high behaved better in both (3B; §7.4.2, Fig. 12).
- **Length.** In a PCA of checkpoint weights from the 24B RL-only run, the authors identify a "length" direction along which mean reward and output length grow together; raw reward scales logarithmically with mean output length between 1500 and 8000 tokens (§7.1, Figs. 8-9).
- **Distillation.** Magistral Small is a distilled-then-RL model; Magistral Medium uses no reasoning SFT. Mistral Medium 3 fine-tuned on about 1.3M open-source (DeepSeek-R1-generated) traces and then trained with RL gains over 10 points on AIME'25 and 5 points on LiveCodeBench, while GPQA Diamond falls from 72.9% to 71.0% (§8, Fig. 13).

## Connections
- [[grpo]] — the base algorithm (Shao et al., 2024) that §2.1 modifies.
- [[magistral-recipe]] — full recipe ledger with loci.
- [[deepseek-r1]] — baseline rows in Table 2; that card records a KL coefficient of 0.001 for R1-Zero, while Magistral removes KL.
- [[dr-grpo]] — Liu et al. (2025), cited in §2.1 as prior GRPO adaptation and in §6.4 for the bias of group normalization.
- [[openr1]] — source of the code-subset prompts used in Magistral Small's cold-start data (§5.3).
- [[llama-4]] — also filters zero-advantage prompts during RL; reports no numbers.
- [[kimi-k1-5]] — reasoning RL with a length penalty and a policy mirror descent objective.
- [[entropy-mechanism-llm-rl]], [[async-rollout]] — entropy collapse in reasoning RL; an open-source asynchronous generator/trainer design.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2506.10910 (v1, PDF full text including Figures 14-16).
- Corrections to the previous card version:
  - "built ... via a ground-up RL pipeline with no distillation" → true for Magistral Medium only; Magistral Small is SFT-cold-started on Magistral Medium traces (Abstract, §5.3).
  - "ε_low ≈ 0.2" → ε_low is not reported (§2.1).
  - "SFT data: not detailed publicly" → sources, difficulty balancing, 10% instruction data, and 4 epochs are disclosed (§5.3); dataset size is not.
  - "Temperature: 0.7 for math/GPQA, 0.95 for code" under training hyperparameters → evaluation settings: temperature 0.7, top-p 1.0 or 0.95 (§5.1); Small's RL temperature is 1.0 (§5.3).
  - "Max completion length 16k → 24k → 32k" → that schedule is for l_max − l_cache; maximum completion length values are not given (§5.2).
  - "Yields ~50% AIME-24 gain" → 26.8 → 73.6 pass@1, called "nearly 50%" (Table 2, §1).
  - "Magistral is text-reasoning-only at this release" → the starting checkpoints are multimodal and multimodal results are reported (§7.2).
  - "Partial/proportional code rewards cost 2% final perf" → 2% lower on LiveCodeBench, 24B, 250 steps (§7.4.1).
  - Missing: Table 3, Table 5, generality results, Small RL settings, data counts → added.
- Removed as unsupported by the source: Guideline "Don't default to KL=0.001 ... KL=0 saves compute and works"; "more compute-efficient and remains stable" (no KL ablation reported); "KL = 0 — atypical; most contemporary GRPO pipelines retain a small KL"; "no distillation from o1/R1/Claude — differentiates from phi-4-reasoning"; "progressive context-length extension drives a self-curriculum"; "Reward-shape breakdown figure" and "batch-size / max-length schedule table" (no such figure or table); "small SFT-only stage"; qwen-3 "Stage-2 GRPO comparison point" and the Kimi "learned CoT RM" contrast (not in this source).
- Not reported by the source: learning rate, optimizer settings, group size G, RL step counts, ε_low, Magistral Medium RL sampling temperature, SFT dataset size, parameter count of Magistral Medium.
