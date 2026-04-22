<!-- scope: reasoning-trace synthesis — HuggingFace open replication of R1 with multi-stage pipeline
     deps: [[deepseek-r1]]
     see-also: [[bespoke-stratos]], [[sky-t1]], [[openr1]], [[openmathinstruct-2]]
-->

# Open-R1: An open-source reproduction of DeepSeek-R1
- **Core Insight:** A fully-open community replication of R1 is feasible by combining (a) mass R1-distilled SFT on a 220K-problem math+code corpus (OpenR1-Math-220k) with (b) GRPO/RLVR on verifiable-reward problems — the HF team published both recipe and data so labs can train R1-quality reasoners without DeepSeek internals.
- **Guideline:** To reproduce R1 open-source: sample 2–8 R1 traces per seed problem, filter each to correct final answer (exact-match or code test), de-dupe and publish as an SFT corpus; then run GRPO with a rule-based verifier on a separate problem pool.
- **Authors:** HuggingFace Open-R1 team (Lewis Tunstall, Edward Beeching, Nathan Lambert, Loubna Ben Allal, Guilherme Penedo, et al.)
- **Year:** 2025
- **URL:** https://github.com/huggingface/open-r1 ; https://huggingface.co/blog/open-r1
- **Relevant topics:** R1 replication, long-CoT SFT, GRPO, RLVR, open data

## Abstract
Open-R1 is HuggingFace's ongoing project to fully reproduce DeepSeek-R1, consisting of three stages: (1) replicate R1-Distill by distilling R1 into open bases with an open dataset; (2) replicate the R1-Zero pure-RL recipe via GRPO with verifiable rewards; (3) replicate the full R1 multi-stage SFT+RL+SFT+RL pipeline. The flagship public artifact is **OpenR1-Math-220k**, a 220K-problem × 2-traces-per-problem math dataset distilled from R1, plus training code and model checkpoints.

## Key Contributions
- **OpenR1-Math-220k** — 220K problems × 2 R1 traces, ~440K samples total, Apache-2.0.
- Open training scripts: SFT, GRPO, distillation, evaluation.
- Continuous model releases: OpenR1-Qwen-7B, OpenR1-Distill-7B.
- Documentation of each replication stage with ablations and negative results.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** 220K math problems aggregated from NuminaMath (cn_k12, olympiads, aops_forum, amc_aime, orca_math), plus supplementary AIME/AMC archives.
- **Trace generation:** query DeepSeek-R1 (HuggingFace-hosted + API) for each problem, sampling 2 traces at temperature 0.6 (some problems sampled up to 8×).
- **Filtering:**
  - Extract boxed answer with regex.
  - Compare to gold using **Math-Verify** (open-source SymPy-based symbolic equivalence checker) — the explicit tool that makes Open-R1's pipeline reproducible.
  - Reject traces with format violations (missing `</think>`, no `\boxed{}`).
- **Output shape:** 220K problems × 2 traces ≈ 440K samples. Long-CoT format preserved. Average trace length ~5K tokens; tail to 30K.
- **Teacher model:** DeepSeek-R1 (671B MoE).
- **Cost / compute:** ~$10K in inference compute (HF's own H100 cluster + API mix); fully documented.

## Modality-specific technical details (REQUIRED — reasoning-trace)
- **Reasoning length distribution:** median ~5K tokens, mean ~7K; ~10% of traces exceed 15K tokens. Significantly longer than OpenMathInstruct-2 (~700 tokens) and Bespoke-Stratos (~3K).
- **Trace style:** R1 native long-CoT — heavy reflection, `<think>…</think>` wrappers. Open-R1 keeps full R1 format in training data.
- **Correctness verifier:** Math-Verify (SymPy-backed) → reliable on algebraic/numeric equivalence but limited on geometry/proofs.
- **GRPO stage (separate from SFT):** uses a binary 0/1 reward from Math-Verify; group-relative advantage with KL penalty; trained on a 40K-problem subset.
- **Error mode:** ~20% of R1's sampled traces fail the verifier; these are simply discarded.

## Quality / diversity evaluation
- OpenR1-Qwen-7B SFT on OpenR1-Math-220k: MATH ~80%, AIME24 ~40%. Close to DeepSeek-R1-Distill-Qwen-7B baselines.
- GRPO stage gives +3–5 AIME points on top of SFT.
- Ablation: 2 traces vs 1 trace per problem → marginal gain; most signal comes from R1's quality not from trace diversity per problem.

## Risks + gotchas
- **R1 dependency:** replication gates on continued availability of R1 weights / API.
- **Math-heavy:** the math-only 220K dataset does not cover code, science, or agentic reasoning. Other tracks in progress.
- **License caveats:** R1 weights are MIT but its output distribution carries "trained by R1" attribution expectations.
- **Verifier limits:** Math-Verify does not catch semantic drift (e.g., answering the wrong question correctly).

## Connections
- Teacher: [[deepseek-r1]].
- Competing/complementary 2025 R1 distill efforts: [[bespoke-stratos]] (17K curated), [[sky-t1]] ($450 recipe).
- SFT corpus lineage: [[numina-math]] (provides the base math pool).
- GRPO ancestry: [[grpo]], [[deepseek-r1]].
- RLVR: [[rlvr-tulu3]].
