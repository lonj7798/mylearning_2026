<!-- scope: Olmo 3 technical report (arXiv:2512.13961v2) RL sections only — §4.4 OlmoRL (algorithm, verifiers, Dolci-Think-RL, infrastructure), §5.4 Instruct RL, §6 RL-Zero, App. A.6.3–A.6.4, Tables 20/23/24/49 — cross-checked against allenai/open-instruct at f020e9a
     deps: [[olmo-3]], [[grpo]], [[rlvr-tulu3]]
     see-also: [[olmo-3-recipe]], [[allenai-olmo3-open-instruct-scripts]], [[open-instruct-rlvr-pipeline]], [[ifbench]], [[truncated-importance-sampling]], [[async-rollout]], [[mai-rl-infrastructure]]
-->

# Olmo 3 — the OlmoRL pipeline (RL sections, cross-checked against open-instruct)
- **Core Insight:** The report locates the RL bottleneck in generation: for the 32B Think run the learner used 8 H100 nodes and inference 20 nodes, the learner spent 75% of the time waiting for data, and the infrastructure gains came from generation-side changes (continuous batching, threading, in-flight weight updates) that raised measured throughput from 881 to 2,949 tokens/s on a fixed 2-node benchmark (§4.4.3, Table 23).
- **Guideline:** For multi-domain RLVR, give each domain its own verifier under one loop, filter out prompts the starting checkpoint already solves (pass rate > 62.5% over 8 samples), and keep the batch full of non-zero-advantage groups by resampling (active sampling) instead of oversampling 3× up front; the report states active sampling reduced loss variance and kept batch size constant (§4.4.2 Step 2, §4.4.3, §6 "Active sampling stabilizes training").
- **Authors:** Olmo Team (Allen Institute for AI; University of Washington; and others) — see [[olmo-3]] for the author list.
- **Year:** 2025 (arXiv v1 2025-12; v2 2026-04-14). Code cross-check: allenai/open-instruct at commit f020e9a9 (2026-11-24).
- **URL:** https://arxiv.org/abs/2512.13961 ; code https://github.com/allenai/open-instruct/tree/f020e9a9a8402a220a67646e3254fc23fa121c27
- **Source type:** official technical report + released code
- **Relevant topics:** RLVR, multi-domain verifiers, LM-judge rewards, active sampling, zero-gradient filtering, truncated importance sampling, continuous batching, in-flight weight updates, RL-Zero, stage order

## Summary
RL is the third post-training stage, after SFT and DPO, for both Think and Instruct; RL-Zero runs RLVR directly on the base model (§4.4, §5.4, §6). OlmoRL is GRPO with seven changes: zero-gradient-signal filtering, active sampling, token-level loss, no KL loss, clip-higher, truncated importance sampling (TIS), and no standard-deviation normalization of advantages (§4.4.1). The prompt set (Dolci-Think-RL) covers math, code, instruction following and general chat, each with its own verifier (§4.4.1 "Verifiers", Figure 16). [[olmo-3]] already covers branch structure, cost, and the 4× headline; this card adds the per-domain data, verifiers, infra numbers, and Table 49 settings.

## Key Contributions
- Five verifier types: math (rule-based normalization + SymPy, binary), code (test cases; pass-rate or all-pass, run on AWS Lambda), IF (constraint functions, "1 if all constraints are satisfied, and 0 otherwise"), chat-with-reference and chat-open-ended (LM judge, score in [0, 1]); judge is Qwen3 32B, thinking off, 32,768-token input, 2,048-token output (§4.4.1, footnotes 35–36, Figure 40).
- Data pipeline in three steps: sourcing, offline difficulty filtering (7B only; 32B relies on active sampling), and mixing with "roughly equal amounts of data … for each domain (with slightly more focus on math and instruction following)" (§4.4.2).
- Fully asynchronous actor/learner design: centralized DeepSpeed learner, independent vLLM actors, prompt queue and results queue (§4.4.3, Figure 17a).
- In-flight weight updates "without pausing the engine … without invalidating the KV cache", following Piché et al. (2025) (§4.4.3).

## Key Figures/Tables to Study
- **Table 20** — Dolci-Think-RL by dataset, with separate columns for Think RL and Instruct RL.
- **Table 23** — infra ablation: OLMo 2 881 tok/s → +continuous batching 975 → +better threading 1,358 → +inflight updates 2,949; MBU 12.90% → 43.21%.
- **Table 24** — IFEval / IFBench by stage (7B Think: SFT 77.9/30.0, DPO 75.9/28.3, RL 88.2/41.6; 32B Think RL 89.0/47.6, 3.1 93.8/68.1).
- **Table 49** — RL hyperparameters for 7B Think, 32B Think, 7B Instruct, 7B RL-Zero.
- **Figure 17** — static vs continuous batching; **Figure 26** — active sampling ablation; **Figure 27** — spurious-reward negative control.

## Technical Details
**Pipeline at a glance** (paper stage → open-instruct file:line at f020e9a)
1. Dataset row (`messages`, `ground_truth`, `dataset` = verifier tag) → `dataset_transformation.py` L1013–1014, `rlvr_tokenize_v1` L2209–2251.
2. Prompt queue, `async_steps` bound ("Max asynchrony") → `data_loader.py` L415, L1412–1430.
3. vLLM actors with continuous batching + prefetch thread → `vllm_utils.py` L435–440.
4. Per-domain verifier dispatch → `ground_truth_utils.py` L1298–1362; verifier classes L245 (math), L333 (IFEval), L920 (code), L744 (LM judge).
5. Zero-std filter + active sampling → `data_loader.py` L817, L1131–1135.
6. Loss: DAPO clip (0.2 / 0.272) × ρ (TIS) weight, token-level mean → `grpo_utils.py` L125–137, L445–496, L539–561; `grpo_fast.py` L720.
7. Weight broadcast thread (NCCL, `inflight_updates`) → `grpo_fast.py` L1479–1513, `grpo_utils.py` L780–817.

**Dolci-Think-RL (Table 20), prompts used in Think RL / Instruct RL.** Precise IF: IF-RLVR 30,186 / 38,000. Math: Open-Reasoner-Zero 3,000 / 14,000; DAPO-Math 2,584 / 7,000; AceReason-Math 6,602 / –; Polaris – / 14,000; KlearReasoner-MathSub 3,000 / 9,000; OMEGA-train 15,000 / 20,000. Code: AceCoder 9,767 / 20,000; KlearReasoner-Code 8,040 / –; Nemotron Post-training Code 2,303 / –; SYNTHETIC-2 3,000 / –. Chat: Tülu 3 SFT 7,129 / 18,955; WildChat-4.8M 7,129 / 18,761; Multi-Subject RLVR 7,129 / 12,234. Totals 104,869 / 171,950.

**Per-domain data processing (§4.4.2).** IF prompts carry up to 5 constraints sampled from IFEval and IFBench-Train ([[ifbench]]). Code: problem rewriting → solution generation → test generation, keep solutions passing > 80% of tests and drop failing tests; AceCoder in function-completion format, others stdio. Chat: Tülu 3 prompts rewritten by GPT-4.1 with extracted references, then filtered by F1 of 8 samples from a Qwen 2.5 7B model; WildChat restricted to English non-reasoning prompts, at most 10 instances per role-play character. Eight OMEGA subtasks downsampled by 50%.

**Infra numbers (§4.4.3).** Rollouts up to 32K tokens, mean > 10K for Think. 32B: 8 learner nodes, 20 inference nodes, ≈5× inference vs training GPU use; 7B: 2 learner nodes, 7 inference nodes, ≈14×. A representative step: 1,000 s, of which 125 s training. Static batching at 32K max and 14,628 mean length would waste up to 54% of compute. 7B Think RL first took ≈15 days without pipelineRL or TIS; the replication on newer infra took 6 days (§4.4.2).

**Table 49 (7B Think / 32B Think / 7B Instruct / 7B RL-Zero).** Dataset 104,869 / 104,869 / 171,950 / 13,314; LR 1e-6 / 2e-6 / 1e-6 / 1e-6, constant; minibatches 1 / 1 / 4 / 1; steps 1,400 / 750 / 450 / 2,000; response length 32,768 / 32,768 / 8,192 / 16,384; unique prompts per batch 64 / 128 / 64 / 32; group size 8; TIS cap – / 2.0 / – / 2.0; temperature 1.0; clip 0.2 / 0.272; learner GPUs 16 / 64 / 8 / 8; actor GPUs 56 / 160 / 56 / 64; TP per actor 1 / 8 / 1 / 1; max asynchrony 1 / 8 / 8 / 8.

**Instruct RL (§5.4).** Same OlmoRL; easier math/code sets, no offline difficulty filtering; max response 8K (7B) and 16K (32B); run from two DPO candidates, final checkpoint chosen by average score, length analysis and a "vibe test". Warm-up on math then mixed data without math was "suboptimal" (footnote 56).

**RL-Zero (§6.1, App. A.6.4).** 13.3K math prompts from DAPO, Klear-Reasoner, Orz, OMEGA after semantic-cluster dedup, decontamination and removal of 8/8-solved prompts; code/IF/chat subsampled from Dolci-Think-RL. Simple per-domain prompt templates (Figure 37) instead of `<think>` templates. Random-reward training produced no gains (Figure 27). RL-Zero 3.1 improved mainly from 16K (not 12K) completions and *not* masking truncated sequences, because masking varied batch size and reduced stability.

**Code cross-check (open-instruct f020e9a).**
```python
# grpo_utils.py L125-137
clip_lower: float = 0.2
clip_higher: float = 0.272
use_rho_correction: bool = True
"""Master switch for the train/infer ratio ρ = π^train_old / π^infer_old correction.
When True, ρ is clamped to [rho_clamp_lower_bound, rho_clamp_upper_bound] ...
This unifies truncated importance sampling ... and IcePop"""
rho_clamp_upper_bound: float = 2.0
```
Clip values and TIS cap 2.0 match Table 49. `7b_think_rl_no_pipeline.sh` sets `--inflight_updates false --async_steps 1` with 16 learner GPUs and 56 vLLM engines (L46–47, L68–69), matching the 7B Think column; the `async_steps` default is 8 (`data_loader.py` L415), matching the other columns. `32b_think_rl.sh` sets `--active_sampling` (L74) and 6 engines × TP 8 (L53–55).

## Verification
- Checked on 2026-11-24 against the arXiv v2 HTML (https://arxiv.org/html/2512.13961v2), §4.4, §5.4, §6, App. A.6, Tables 20/23/24/49, and against the local clone of open-instruct at f020e9a.
- Discrepancy, paper vs code: the paper's IF verifier is binary (all constraints satisfied → 1); `IFEvalVerifier` at f020e9a returns the fraction of satisfied constraints, `sum(rewards) / max(len(rewards), 1)` (`ground_truth_utils.py` L381). Not resolved which version trained the released models.
- Discrepancy, existing card: [[olmo-3]] reads Table 20 as "104,869 prompts for 7B and 171,950 for 32B". Table 20's columns are Think RL vs Instruct RL, and Table 49 lists 104,869 for both 7B and 32B Think RL and 171,950 for 7B Instruct RL. That card is outside this library and was not edited.
- Equation (1) and (2) bodies were not extractable from the HTML (math rendered as images); only their textual description is used above.
- Not reported: per-domain reward weights in the final mix (only "roughly equal"); per-domain retention or interference numbers.
